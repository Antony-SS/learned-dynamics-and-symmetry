"""IDSIA physics and residual-physics baselines, without pytorch3d.

Reimplements Busetto et al. PhysQuadModel / ResidualQuadModel / PhysResQuadModel
so we can load their published checkpoints and roll them with our integrator.
We do not retrain these baselines: physics+residual is always the authors'
released ``phys+res_random_square_chirp.pt``.

Their published PhysQuadModel multiplies omega_dot by 0 (angular velocity is
held at the window seed while thrust still integrates). Physics-only and the
frozen physics inside physics+residual both do that, so Melon MAE matches
their exported ``physics_model_multistep`` predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from quadrotor.flight import PreparedFlight
from quadrotor.geometry import so3_log_batch
from quadrotor.idsia.data.load import DEFAULT_ROOT
from quadrotor.geometry import _so3_exp, _so3_log

DT = 0.01
PHYS_RES_NAME = "phys+res_random_square_chirp.pt"
RESIDUAL_NAME = "residual_random_square_chirp.pt"
CRAZYFLIE_PARAMS = {
    "g": 9.81,
    "m": 0.045,
    "J": torch.diag(torch.tensor([2.3951e-5, 2.3951e-5, 3.2347e-6])),
    "thrust_to_weight": 2.0,
    "max_torque": torch.tensor([1e-2, 1e-2, 3e-3]),
}


def _checkpoint_dir(root: Path | None = None) -> Path:
    return (DEFAULT_ROOT if root is None else Path(root)) / "out" / "models"


class BaseQuadModel(torch.nn.Module):
    """One-step state map x,u -> x_next, plus the open-loop integrator hook."""

    reads_pose = True

    def __init__(self, dt: float = DT) -> None:
        super().__init__()
        self.dt = float(dt)

    def one_step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def step_real(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.one_step(x, u)

    def initial_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(batch_size, 0)

    def advance(
        self,
        position: torch.Tensor,
        rotation: torch.Tensor,
        velocity_world: torch.Tensor,
        omega_body: torch.Tensor,
        motors: torch.Tensor,
        dt: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del dt
        so3 = _so3_log(rotation)
        x = torch.cat([position, velocity_world, so3, omega_body], dim=-1)
        x_next = self.step_real(x, motors)
        position, velocity_world, so3, omega_body = x_next.split((3, 3, 3, 3), dim=-1)
        return position, _so3_exp(so3), velocity_world, omega_body, hidden

    def forward(self, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        if u_seq.ndim == 2:
            u_seq = u_seq.unsqueeze(1)
        x = x0.squeeze(1) if x0.ndim == 3 else x0
        preds = []
        for index in range(u_seq.shape[1]):
            x = self.one_step(x, u_seq[:, index])
            preds.append(x.unsqueeze(1))
        return torch.cat(preds, dim=1)


class PhysQuadModel(BaseQuadModel):
    """RK4 rigid-body Crazyflie. State is [pos, vel, so3, omega], motors in rad/s."""

    def __init__(
        self,
        params: dict | None = None,
        dt: float = DT,
        arm_length: float = 0.0353,
        kt: float = 3.72e-08,
        kc: float = 7.74e-12,
        zero_omega_dot: bool = False,
    ) -> None:
        super().__init__(dt)
        params = CRAZYFLIE_PARAMS if params is None else params
        self.m = float(params["m"])
        self.g = float(params["g"])
        self.thrust_to_weight = float(params["thrust_to_weight"])
        self.arm = float(arm_length)
        self.kt = float(kt)
        self.kc = float(kc)
        self.t_max = self.thrust_to_weight * self.m * self.g
        self.zero_omega_dot = bool(zero_omega_dot)
        self.register_buffer("J", torch.as_tensor(params["J"], dtype=torch.float32))
        self.register_buffer("J_inv", torch.linalg.inv(self.J))
        self.register_buffer(
            "max_torque",
            torch.as_tensor(params["max_torque"], dtype=torch.float32).view(-1),
        )
        self.register_buffer(
            "gravity", torch.tensor([0.0, 0.0, self.g], dtype=torch.float32)
        )

    @torch.no_grad()
    def motor_to_phys(self, motors: torch.Tensor) -> torch.Tensor:
        omega2 = motors ** 2
        thrust = self.kt * omega2.sum(dim=1)
        tau_x = self.kt * self.arm * (
            (omega2[:, 2] + omega2[:, 3]) - (omega2[:, 0] + omega2[:, 1])
        )
        tau_y = self.kt * self.arm * (
            (omega2[:, 1] + omega2[:, 2]) - (omega2[:, 0] + omega2[:, 3])
        )
        tau_z = self.kc * ((omega2[:, 0] + omega2[:, 2]) - (omega2[:, 1] + omega2[:, 3]))
        t_norm = thrust / self.t_max
        tau_norm = torch.stack([tau_x, tau_y, tau_z], dim=1) / self.max_torque
        return torch.cat([t_norm.unsqueeze(1), tau_norm], dim=1)

    def rates(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        quat: torch.Tensor,
        omega: torch.Tensor,
        u_mot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Continuous rigid-body rates: ṗ, v̇, q̇, ω̇. Does not integrate."""
        u_phys = self.motor_to_phys(u_mot)
        t_norm = torch.clamp(u_phys[:, 0], 0.0, 1.0)
        tau_norm = torch.clamp(u_phys[:, 1:], -1.0, 1.0)
        thrust = t_norm * self.t_max
        tau = tau_norm * self.max_torque
        thrust_body = torch.zeros_like(vel)
        thrust_body[:, 2] = thrust
        acc = (quat_rotate(quat, thrust_body) - self.m * self.gravity) / self.m
        j_omega = omega @ self.J.T
        omega_dot = torch.linalg.solve(
            self.J, (tau - torch.cross(omega, j_omega, dim=-1)).unsqueeze(-1)
        ).squeeze(-1)
        if self.zero_omega_dot:
            omega_dot = omega_dot * 0.0
        return vel, acc, quat_derivative(quat, omega), omega_dot

    def one_step(self, x: torch.Tensor, u_mot: torch.Tensor) -> torch.Tensor:
        u_phys = self.motor_to_phys(u_mot)
        pos, vel, so3, omega = x.split((3, 3, 3, 3), dim=-1)
        quat = so3_to_quat_xyzw(so3)
        next_quat_state = self._step_from_phys(
            torch.cat([pos, vel, quat, omega], dim=-1), u_phys
        )
        pos, vel, quat, omega = next_quat_state.split((3, 3, 4, 3), dim=-1)
        return torch.cat([pos, vel, quat_xyzw_to_so3(quat), omega], dim=-1)

    def _step_from_phys(self, x: torch.Tensor, u_phys: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        pos, vel, quat, omega = x.split((3, 3, 4, 3), dim=-1)

        def dynamics(pos, vel, quat, omega):
            t_norm = torch.clamp(u_phys[:, 0], 0.0, 1.0)
            tau_norm = torch.clamp(u_phys[:, 1:], -1.0, 1.0)
            thrust = t_norm * self.t_max
            tau = tau_norm * self.max_torque
            thrust_body = torch.zeros_like(vel)
            thrust_body[:, 2] = thrust
            acc = (quat_rotate(quat, thrust_body) - self.m * self.gravity) / self.m
            j_omega = omega @ self.J.T.to(dtype=omega.dtype)
            omega_dot = torch.linalg.solve(
                self.J, (tau - torch.cross(omega, j_omega, dim=-1)).unsqueeze(-1)
            ).squeeze(-1)
            if self.zero_omega_dot:
                omega_dot = omega_dot * 0.0
            return vel, acc, quat_derivative(quat, omega), omega_dot

        v1, a1, qd1, w1 = dynamics(pos, vel, quat, omega)
        pos2 = pos + 0.5 * dt * v1
        vel2 = vel + 0.5 * dt * a1
        quat2 = F.normalize(quat + 0.5 * dt * qd1, dim=-1)
        omega2 = omega + 0.5 * dt * w1
        v2, a2, qd2, w2 = dynamics(pos2, vel2, quat2, omega2)
        pos3 = pos + 0.5 * dt * v2
        vel3 = vel + 0.5 * dt * a2
        quat3 = F.normalize(quat + 0.5 * dt * qd2, dim=-1)
        omega3 = omega + 0.5 * dt * w2
        v3, a3, qd3, w3 = dynamics(pos3, vel3, quat3, omega3)
        pos4 = pos + dt * v3
        vel4 = vel + dt * a3
        quat4 = F.normalize(quat + dt * qd3, dim=-1)
        omega4 = omega + dt * w3
        v4, a4, qd4, w4 = dynamics(pos4, vel4, quat4, omega4)
        pos_next = pos + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        vel_next = vel + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        omega_next = omega + (dt / 6.0) * (w1 + 2 * w2 + 2 * w3 + w4)
        quat_next = F.normalize(quat + (dt / 6.0) * (qd1 + 2 * qd2 + 2 * qd3 + qd4), dim=-1)
        return torch.cat([pos_next, vel_next, quat_next, omega_next], dim=-1)


class ResidualQuadModel(BaseQuadModel):
    """MLP residual step dx on (x, u). Hidden 64, five layers matches their checkpoint."""

    def __init__(
        self,
        state_dim: int = 12,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 5,
        dt: float = DT,
    ) -> None:
        super().__init__(dt)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        layers: list[torch.nn.Module] = []
        dim = state_dim + input_dim
        for index in range(num_layers):
            layers += [torch.nn.Linear(dim if index == 0 else hidden_dim, hidden_dim), torch.nn.ReLU()]
        self.mlp = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(hidden_dim, state_dim)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)
        self.uses_scalers = False

    def set_scalers(
        self,
        x_mean: torch.Tensor,
        x_scale: torch.Tensor,
        u_mean: torch.Tensor,
        u_scale: torch.Tensor,
    ) -> None:
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32))
        self.register_buffer(
            "x_scale", torch.as_tensor(x_scale, dtype=torch.float32).clamp_min(1e-6)
        )
        self.register_buffer("u_mean", torch.as_tensor(u_mean, dtype=torch.float32))
        self.register_buffer(
            "u_scale", torch.as_tensor(u_scale, dtype=torch.float32).clamp_min(1e-6)
        )
        self.uses_scalers = True

    def one_step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + self.out(self.mlp(torch.cat([x, u], dim=-1)))

    def step_real(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if not self.uses_scalers:
            return self.one_step(x, u)
        x_n = (x - self.x_mean) / self.x_scale
        u_n = (u - self.u_mean) / self.u_scale
        return self.one_step(x_n, u_n) * self.x_scale + self.x_mean


class PhysResQuadModel(BaseQuadModel):
    """Frozen physics plus an MLP residual in normalized coordinates."""

    def __init__(self, phys: PhysQuadModel, residual: ResidualQuadModel) -> None:
        super().__init__(phys.dt)
        self.phys = phys
        self.neural = residual
        self.register_buffer("x_mean", torch.zeros(12))
        self.register_buffer("x_scale", torch.ones(12))
        self.register_buffer("u_mean", torch.zeros(4))
        self.register_buffer("u_scale", torch.ones(4))

    def x_denorm(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.x_scale + self.x_mean

    def x_normed(self, x_real: torch.Tensor) -> torch.Tensor:
        return (x_real - self.x_mean) / self.x_scale

    def u_denorm(self, u_norm: torch.Tensor) -> torch.Tensor:
        return u_norm * self.u_scale + self.u_mean

    def one_step(self, x_norm: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        x_real = self.x_denorm(x_norm)
        u_mot = self.u_denorm(u_norm)
        with torch.no_grad():
            x_phys_next = self.x_normed(self.phys.one_step(x_real, u_mot))
        dx = self.neural.out(self.neural.mlp(torch.cat([x_norm, u_norm], dim=-1)))
        x_next = x_phys_next + dx
        if not torch.all(torch.isfinite(x_next)):
            return x_phys_next
        return x_next

    def step_real(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        u_n = (u - self.u_mean) / self.u_scale
        return self.x_denorm(self.one_step(self.x_normed(x), u_n))



def so3_to_quat_xyzw(rotation_vector: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotation_vector, dim=-1, keepdim=True)
    half = 0.5 * theta
    small = theta < 1e-8
    axis = rotation_vector / theta.clamp_min(1e-8)
    xyz = torch.where(small, 0.5 * rotation_vector, axis * torch.sin(half))
    w = torch.where(small, torch.ones_like(theta), torch.cos(half))
    return torch.cat([xyz, w], dim=-1)


def quat_xyzw_to_so3(quat: torch.Tensor) -> torch.Tensor:
    quat = F.normalize(quat, dim=-1)
    xyz, w = quat[..., :3], quat[..., 3]
    flip = w < 0
    xyz = torch.where(flip.unsqueeze(-1), -xyz, xyz)
    w = torch.where(flip, -w, w)
    sine = torch.linalg.norm(xyz, dim=-1)
    theta = 2.0 * torch.atan2(sine, w)
    small = sine < 1e-8
    axis = xyz / sine.clamp_min(1e-8).unsqueeze(-1)
    return torch.where(small.unsqueeze(-1), 2.0 * xyz, axis * theta.unsqueeze(-1))


def quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    qv, qw = quat[:, :3], quat[:, 3:4]
    temp = 2.0 * torch.cross(qv, vector, dim=-1)
    return vector + qw * temp + torch.cross(qv, temp, dim=-1)


def quat_derivative(quat: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    qv, qw = quat[:, :3], quat[:, 3:4]
    dqv = 0.5 * (qw * omega + torch.cross(qv, omega, dim=-1))
    dqw = -0.5 * (qv * omega).sum(dim=-1, keepdim=True)
    return torch.cat([dqv, dqw], dim=-1)


def _unpack_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"unrecognized checkpoint type {type(checkpoint)}")


def load_physics(device: str | torch.device = "cpu") -> PhysQuadModel:
    """IDSIA PhysQuadModel, including the published ``omega_dot = 0`` step."""
    model = PhysQuadModel(zero_omega_dot=True)
    model.to(device)
    model.eval()
    return model


def load_phys_res(
    root: Path | None = None,
    device: str | torch.device = "cpu",
) -> PhysResQuadModel:
    path = _checkpoint_dir(root) / PHYS_RES_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    phys = PhysQuadModel(zero_omega_dot=True)
    residual = ResidualQuadModel(hidden_dim=64, num_layers=5)
    model = PhysResQuadModel(phys, residual)
    model.load_state_dict(_unpack_state(checkpoint), strict=True)
    model.to(device)
    model.eval()
    return model


def load_residual(
    root: Path | None = None,
    device: str | torch.device = "cpu",
    train_flights: list[PreparedFlight] | None = None,
) -> ResidualQuadModel:
    path = _checkpoint_dir(root) / RESIDUAL_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = ResidualQuadModel(
        hidden_dim=int(config.get("hidden_dim") or 64),
        num_layers=int(config.get("num_layers") or 5),
    )
    state = _unpack_state(checkpoint)
    # Their residual checkpoint is only the MLP. Scalers were sklearn pickles
    # that were never committed; refit mean/std on the training flights.
    model.load_state_dict(state, strict=True)
    if train_flights:
        x_mean, x_scale, u_mean, u_scale = fit_scalers(train_flights)
        model.set_scalers(x_mean, x_scale, u_mean, u_scale)
    model.to(device)
    model.eval()
    return model


def fit_scalers(
    flights: list[PreparedFlight],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = []
    motors = []
    for flight in flights:
        so3 = so3_log_batch(flight.rotation).astype(np.float32)
        states.append(
            np.concatenate(
                [flight.position, flight.velocity_world, so3, flight.omega_body],
                axis=1,
            )
        )
        motors.append(flight.motors)
    x = np.concatenate(states, axis=0)
    u = np.concatenate(motors, axis=0)
    x_scale = np.std(x, axis=0).astype(np.float32)
    u_scale = np.std(u, axis=0).astype(np.float32)
    return (
        torch.as_tensor(np.mean(x, axis=0), dtype=torch.float32),
        torch.as_tensor(np.clip(x_scale, 1e-6, None), dtype=torch.float32),
        torch.as_tensor(np.mean(u, axis=0), dtype=torch.float32),
        torch.as_tensor(np.clip(u_scale, 1e-6, None), dtype=torch.float32),
    )


def fit_world_scalers(
    flights: list[PreparedFlight],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean / std of 16-D world features ``(log R, p, v, ω, Ω)``."""
    rows = []
    for flight in flights:
        so3 = so3_log_batch(flight.rotation[:-1]).astype(np.float32)
        rows.append(
            np.concatenate(
                [
                    so3,
                    flight.position[:-1],
                    flight.velocity_world[:-1],
                    flight.omega_body[:-1],
                    flight.motors,
                ],
                axis=1,
            )
        )
    values = np.concatenate(rows, axis=0)
    scale = np.clip(np.std(values, axis=0).astype(np.float32), 1e-6, None)
    return (
        torch.as_tensor(np.mean(values, axis=0), dtype=torch.float32),
        torch.as_tensor(scale, dtype=torch.float32),
    )


def fit_canonical_scalers(
    flights: list[PreparedFlight],
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for flight in flights:
        rotation = flight.rotation[:-1]
        up = np.broadcast_to(np.asarray((0.0, 0.0, 1.0), dtype=np.float32), (len(rotation), 3))
        gravity_body = np.einsum("nji,nj->ni", rotation, up)
        velocity_body = np.einsum("nji,nj->ni", rotation, flight.velocity_world[:-1])
        rows.append(
            np.concatenate(
                [gravity_body, velocity_body, flight.omega_body[:-1], flight.motors],
                axis=1,
            )
        )
    values = np.concatenate(rows, axis=0)
    scale = np.clip(np.std(values, axis=0).astype(np.float32), 1e-6, None)
    return (
        torch.as_tensor(np.mean(values, axis=0), dtype=torch.float32),
        torch.as_tensor(scale, dtype=torch.float32),
    )


def _windows(
    flights: list[PreparedFlight],
    horizon: int,
    stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stride = horizon if stride is None else int(stride)
    xs, us, ys = [], [], []
    for flight in flights:
        so3 = so3_log_batch(flight.rotation).astype(np.float32)
        state = np.concatenate(
            [flight.position, flight.velocity_world, so3, flight.omega_body],
            axis=1,
        )
        last = flight.n_steps - horizon + 1
        for start in range(0, last, stride):
            stop = start + horizon
            xs.append(state[start])
            us.append(flight.motors[start:stop])
            ys.append(state[start + 1 : stop + 1])
    return (
        torch.as_tensor(np.stack(xs), dtype=torch.float32),
        torch.as_tensor(np.stack(us), dtype=torch.float32),
        torch.as_tensor(np.stack(ys), dtype=torch.float32),
    )
