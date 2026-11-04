"""The three IDSIA models we train.

QuadDynamicsNet: delay-line MLP on the 13-D quotient → body (a, α), then kinematics.
PhysQuotient: frozen Crazyflie RK4 + 12-D residual that reads the quotient only.
PhysQuotientSym: same physics, body-frame residual with Z₂×Z₂ orbit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quadrotor.idsia.baselines import BaseQuadModel, PhysQuadModel, ResidualQuadModel
from quadrotor.train import _apply, _features, _so3_exp, _so3_log

HISTORY = 10
HIDDEN = 64

_ORBIT_GV = (
    (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0, 1.0, -1.0, 1.0),
    (-1.0, 1.0, 1.0, -1.0, 1.0, 1.0),
    (-1.0, -1.0, 1.0, -1.0, -1.0, 1.0),
)
_ORBIT_W = (
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, -1.0),
    (1.0, -1.0, -1.0),
    (-1.0, -1.0, 1.0),
)
_ORBIT_PWM = ((0, 1, 2, 3), (3, 2, 1, 0), (1, 0, 3, 2), (2, 3, 0, 1))
_ORBIT_ACC = (
    (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
    (-1.0, 1.0, 1.0, 1.0, -1.0, -1.0),
    (-1.0, -1.0, 1.0, -1.0, -1.0, 1.0),
)


def _mlp(in_dim: int, hidden: int, n_hidden: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = int(in_dim)
    for _ in range(int(n_hidden)):
        layers += [nn.Linear(dim, hidden), nn.ReLU()]
        dim = int(hidden)
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


def quotient(rotation: torch.Tensor, velocity_world: torch.Tensor, omega: torch.Tensor, pwm: torch.Tensor) -> torch.Tensor:
    dummy = type("M", (), {"reads_pose": False, "reads_voltage": False, "reads_height": False})()
    return _features(dummy, None, rotation, velocity_world, omega, pwm, None)


class QuadDynamicsNet(nn.Module):
    """Stacked quotient frames → body linear and angular acceleration.

    Optional Z₂×Z₂ orbit average (default on). Integrator is kinematics only.
    """

    reads_pose = False
    reads_voltage = False
    reads_height = False
    family = "quad_dynamics"

    def __init__(self, symmetric: bool = True, hidden: int = HIDDEN, n_hidden: int = 2) -> None:
        super().__init__()
        self.history = HISTORY
        self.hidden = int(hidden)
        self.n_hidden = int(n_hidden)
        self.symmetric = bool(symmetric)
        self.net = _mlp(13 * self.history, self.hidden, self.n_hidden, 6)
        self.register_buffer("x_mean", torch.zeros(13))
        self.register_buffer("x_scale", torch.ones(13))
        self.register_buffer("orbit_gv", torch.tensor(_ORBIT_GV, dtype=torch.float32))
        self.register_buffer("orbit_w", torch.tensor(_ORBIT_W, dtype=torch.float32))
        self.register_buffer("orbit_pwm", torch.tensor(_ORBIT_PWM, dtype=torch.long))
        self.register_buffer("orbit_acc", torch.tensor(_ORBIT_ACC, dtype=torch.float32))

    def set_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        with torch.no_grad():
            self.x_mean.copy_(mean.reshape(13).to(self.x_mean))
            self.x_scale.copy_(scale.reshape(13).to(self.x_scale).clamp_min(1e-6))

    def _predict(self, stacked: torch.Tensor) -> torch.Tensor:
        if not self.symmetric:
            return self.net(stacked)
        batch = stacked.shape[0]
        slots = stacked.reshape(batch, self.history, 13)
        pwm = slots[..., 9:13]
        index = self.orbit_pwm.view(4, 1, 1, 4).expand(4, batch, self.history, 4)
        orbit = torch.cat(
            (
                slots[..., 0:6].unsqueeze(0) * self.orbit_gv.view(4, 1, 1, 6),
                slots[..., 6:9].unsqueeze(0) * self.orbit_w.view(4, 1, 1, 3),
                pwm.unsqueeze(0).expand(4, -1, -1, -1).gather(-1, index),
            ),
            dim=-1,
        )
        raw = self.net(orbit.reshape(4 * batch, -1)).reshape(4, batch, 6)
        return (raw * self.orbit_acc.unsqueeze(1)).sum(0) / 4.0

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        return self._predict(stacked)

    def initial_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(batch_size, self.history, 13)

    def step(self, features: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        buf = torch.cat((hidden[:, 1:], features.unsqueeze(1)), dim=1)
        flat = ((buf - self.x_mean) / self.x_scale).reshape(buf.shape[0], -1)
        return self._predict(flat), buf

    def advance(self, p, r, v_w, w, pwm, dt, hidden):
        feat = quotient(r, v_w, w, pwm)
        acc, hidden = self.step(feat, hidden)
        p, r, v_w, w = _apply(p, r, v_w, w, acc, dt)
        return p, r, v_w, w, hidden


class PhysQuotient(BaseQuadModel):
    """Frozen IDSIA physics + 12-D residual on the 13-D quotient (no p, no yaw)."""

    family = "phys_quotient"

    def __init__(self) -> None:
        super().__init__()
        self.phys = PhysQuadModel(zero_omega_dot=True)
        self.neural = ResidualQuadModel(state_dim=12, input_dim=1, hidden_dim=64, num_layers=5)
        self.neural.mlp = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.neural.out = nn.Linear(64, 12)
        nn.init.zeros_(self.neural.out.weight)
        nn.init.zeros_(self.neural.out.bias)
        self.register_buffer("x_mean", torch.zeros(12))
        self.register_buffer("x_scale", torch.ones(12))
        self.register_buffer("u_mean", torch.zeros(4))
        self.register_buffer("u_scale", torch.ones(4))
        self.register_buffer("canonical_mean", torch.zeros(13))
        self.register_buffer("canonical_scale", torch.ones(13))

    def x_denorm(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.x_scale + self.x_mean

    def x_normed(self, x_real: torch.Tensor) -> torch.Tensor:
        return (x_real - self.x_mean) / self.x_scale

    def _canonical(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        _pos, vel, so3, omega = x.split((3, 3, 3, 3), dim=-1)
        rotation_t = _so3_exp(so3).transpose(-1, -2)
        up = x.new_tensor((0.0, 0.0, 1.0)).expand(x.shape[0], 3)
        gravity_body = torch.bmm(rotation_t, up.unsqueeze(-1)).squeeze(-1)
        velocity_body = torch.bmm(rotation_t, vel.unsqueeze(-1)).squeeze(-1)
        return torch.cat((gravity_body, velocity_body, omega, u), dim=-1)

    def one_step(self, x_norm: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        x_real = self.x_denorm(x_norm)
        u_mot = u_norm * self.u_scale + self.u_mean
        with torch.no_grad():
            x_phys_next = self.x_normed(self.phys.one_step(x_real, u_mot))
        canon = (self._canonical(x_real, u_mot) - self.canonical_mean) / self.canonical_scale
        dx = self.neural.out(self.neural.mlp(canon))
        x_next = x_phys_next + dx
        if not torch.all(torch.isfinite(x_next)):
            return x_phys_next
        return x_next

    def step_real(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        u_n = (u - self.u_mean) / self.u_scale
        return self.x_denorm(self.one_step(self.x_normed(x), u_n))


class SymmetricResidual(nn.Module):
    """13-D quotient → body (Δp, Δv, Δθ, Δω), Reynolds-averaged over four mirrors."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(num_layers):
            layers += [nn.Linear(13 if index == 0 else hidden_dim, hidden_dim), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, 12)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("orbit_gv", torch.tensor(_ORBIT_GV, dtype=torch.float32))
        self.register_buffer("orbit_w", torch.tensor(_ORBIT_W, dtype=torch.float32))
        self.register_buffer("orbit_pwm", torch.tensor(_ORBIT_PWM, dtype=torch.long))
        self.register_buffer(
            "orbit_polar",
            torch.tensor(
                ((1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (-1.0, 1.0, 1.0), (-1.0, -1.0, 1.0)),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "orbit_axial",
            torch.tensor(
                ((1.0, 1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, -1.0, -1.0), (-1.0, -1.0, 1.0)),
                dtype=torch.float32,
            ),
        )

    def body_dx(self, canonical: torch.Tensor) -> torch.Tensor:
        batch = canonical.shape[0]
        pwm = canonical[:, 9:13]
        index = self.orbit_pwm.view(4, 1, 4).expand(4, batch, 4)
        orbit = torch.cat(
            (
                canonical[:, 0:6].unsqueeze(0) * self.orbit_gv.view(4, 1, 6),
                canonical[:, 6:9].unsqueeze(0) * self.orbit_w.view(4, 1, 3),
                pwm.unsqueeze(0).expand(4, -1, -1).gather(-1, index),
            ),
            dim=-1,
        )
        raw = self.out(self.mlp(orbit.reshape(4 * batch, 13))).reshape(4, batch, 12)
        polar = self.orbit_polar.view(4, 1, 3)
        axial = self.orbit_axial.view(4, 1, 3)
        signed = torch.cat(
            (
                raw[..., 0:3] * polar,
                raw[..., 3:6] * polar,
                raw[..., 6:9] * axial,
                raw[..., 9:12] * axial,
            ),
            dim=-1,
        )
        return signed.mean(dim=0)


class PhysQuotientSym(BaseQuadModel):
    """Frozen IDSIA physics + body 12-D residual with Z₂×Z₂ orbit."""

    family = "phys_quotient_sym"

    def __init__(self) -> None:
        super().__init__()
        self.phys = PhysQuadModel(zero_omega_dot=True)
        self.neural = SymmetricResidual(hidden_dim=64, num_layers=5)
        self.register_buffer("canonical_mean", torch.zeros(13))
        self.register_buffer("canonical_scale", torch.ones(13))

    def _canonical(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        _pos, vel, so3, omega = x.split((3, 3, 3, 3), dim=-1)
        rotation = _so3_exp(so3)
        rotation_t = rotation.transpose(-1, -2)
        up = x.new_tensor((0.0, 0.0, 1.0)).expand(x.shape[0], 3)
        gravity_body = torch.bmm(rotation_t, up.unsqueeze(-1)).squeeze(-1)
        velocity_body = torch.bmm(rotation_t, vel.unsqueeze(-1)).squeeze(-1)
        return torch.cat((gravity_body, velocity_body, omega, u), dim=-1)

    def one_step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_phys = self.phys.one_step(x, u)
        canonical = (self._canonical(x, u) - self.canonical_mean) / self.canonical_scale
        dx_body = self.neural.body_dx(canonical)
        delta_pos, delta_vel, delta_rot, delta_omega = dx_body.split((3, 3, 3, 3), dim=-1)
        pos_p, vel_p, so3_p, omega_p = x_phys.split((3, 3, 3, 3), dim=-1)
        rotation = _so3_exp(x[:, 6:9])
        pos_next = pos_p + torch.bmm(rotation, delta_pos.unsqueeze(-1)).squeeze(-1)
        vel_next = vel_p + torch.bmm(rotation, delta_vel.unsqueeze(-1)).squeeze(-1)
        rotation_next = torch.bmm(_so3_exp(so3_p), _so3_exp(delta_rot))
        omega_next = omega_p + delta_omega
        return torch.cat((pos_next, vel_next, _so3_log(rotation_next), omega_next), dim=-1)


def make_model(family: str, symmetric: bool = True) -> nn.Module:
    if family == "quad_dynamics":
        return QuadDynamicsNet(symmetric=symmetric)
    if family == "phys_quotient":
        return PhysQuotient()
    if family == "phys_quotient_sym":
        return PhysQuotientSym()
    raise ValueError(f"unknown family {family}")
