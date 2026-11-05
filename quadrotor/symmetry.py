"""Quotient features, Z₂×Z₂ orbit, and the body increment lift.

Tables live here once. Residual MLPs stay geometry-free: they write body
``(Δp, Δv, Δθ, Δω)``. Polar parts are lifted by the current attitude ``R``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quadrotor.geometry import _features, _so3_exp, _so3_log

# Four Klein-group mirrors. Polar (g, v, Δp, Δv) vs axial (ω, Δθ, Δω).
# Motor slots stay Crazyflie-X order; the last three rows permute them.
ORBIT_GV = (
    (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0, 1.0, -1.0, 1.0),
    (-1.0, 1.0, 1.0, -1.0, 1.0, 1.0),
    (-1.0, -1.0, 1.0, -1.0, -1.0, 1.0),
)
ORBIT_W = (
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, -1.0),
    (1.0, -1.0, -1.0),
    (-1.0, -1.0, 1.0),
)
ORBIT_MOTORS = ((0, 1, 2, 3), (3, 2, 1, 0), (1, 0, 3, 2), (2, 3, 0, 1)) # Crazyflie-X motor order, then permuted by Z₂×Z₂ across xz-plane and yz-plane
ORBIT_POLAR = (
    (1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0),
    (-1.0, 1.0, 1.0),
    (-1.0, -1.0, 1.0),
)
ORBIT_AXIAL = (
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, -1.0),
    (1.0, -1.0, -1.0),
    (-1.0, -1.0, 1.0),
)


def _as_orbit_tensor(table, device, dtype, long: bool = False) -> torch.Tensor:
    return torch.tensor(table, device=device, dtype=torch.long if long else dtype)


def quotient(
    rotation: torch.Tensor,
    velocity_world: torch.Tensor,
    omega: torch.Tensor,
    motors: torch.Tensor,
) -> torch.Tensor:
    """13-D body features ``(g_body, v_body, ω, Ω)``. Drops ``p`` and yaw."""
    return _features(rotation, velocity_world, omega, motors)


def world_features(
    position: torch.Tensor,
    rotation: torch.Tensor,
    velocity_world: torch.Tensor,
    omega: torch.Tensor,
    motors: torch.Tensor,
) -> torch.Tensor:
    """16-D ``(log R, p, v, ω, Ω)`` in the world / so(3) chart."""
    return torch.cat((_so3_log(rotation), position, velocity_world, omega, motors), dim=-1)


def features_from_state(x: torch.Tensor, u: torch.Tensor, kind: str) -> torch.Tensor:
    pos, vel, so3, omega = x.split((3, 3, 3, 3), dim=-1)
    rotation = _so3_exp(so3)
    if kind == "world":
        return world_features(pos, rotation, vel, omega, u)
    if kind == "quotient":
        return quotient(rotation, vel, omega, u)
    raise ValueError(f"unknown features {kind}")


def apply_orbit_element(canonical: torch.Tensor, index: int) -> torch.Tensor:
    """Apply one Z₂×Z₂ element to a 13-D quotient row."""
    gv = canonical[..., 0:6] * canonical.new_tensor(ORBIT_GV[index])
    omega = canonical[..., 6:9] * canonical.new_tensor(ORBIT_W[index])
    motors = canonical[..., 9:13][..., list(ORBIT_MOTORS[index])]
    return torch.cat((gv, omega, motors), dim=-1)


def orbit_features(canonical: torch.Tensor) -> torch.Tensor:
    """Stack the four mirrors. Shape ``(4, B, 13)``."""
    batch = canonical.shape[0]
    device, dtype = canonical.device, canonical.dtype
    gv = _as_orbit_tensor(ORBIT_GV, device, dtype)
    omega = _as_orbit_tensor(ORBIT_W, device, dtype)
    motors_idx = _as_orbit_tensor(ORBIT_MOTORS, device, dtype, long=True)
    motors = canonical[:, 9:13]
    index = motors_idx.view(4, 1, 4).expand(4, batch, 4)
    return torch.cat(
        (
            canonical[:, 0:6].unsqueeze(0) * gv.view(4, 1, 6),
            canonical[:, 6:9].unsqueeze(0) * omega.view(4, 1, 3),
            motors.unsqueeze(0).expand(4, -1, -1).gather(-1, index),
        ),
        dim=-1,
    )


def orbit_body_dx(raw: torch.Tensor) -> torch.Tensor:
    """Sign-correct a ``(4, B, 12)`` body increment and Reynolds-average."""
    polar = raw.new_tensor(ORBIT_POLAR).view(4, 1, 3)
    axial = raw.new_tensor(ORBIT_AXIAL).view(4, 1, 3)
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


def apply_body_dx(x_phys: torch.Tensor, x: torch.Tensor, dx_body: torch.Tensor) -> torch.Tensor:
    """Add a body increment after RK4, lifting polar parts through current ``R``."""
    delta_pos, delta_vel, delta_rot, delta_omega = dx_body.split((3, 3, 3, 3), dim=-1)
    pos_p, vel_p, so3_p, omega_p = x_phys.split((3, 3, 3, 3), dim=-1)
    rotation = _so3_exp(x[:, 6:9])
    pos_next = pos_p + torch.bmm(rotation, delta_pos.unsqueeze(-1)).squeeze(-1)
    vel_next = vel_p + torch.bmm(rotation, delta_vel.unsqueeze(-1)).squeeze(-1)
    rotation_next = torch.bmm(_so3_exp(so3_p), _so3_exp(delta_rot))
    omega_next = omega_p + delta_omega
    return torch.cat((pos_next, vel_next, _so3_log(rotation_next), omega_next), dim=-1)


class Quotient(nn.Module):
    """World ``(p, R, v, ω, Ω)`` packed as ``x, u`` → 13-D quotient."""

    dim = 13

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return features_from_state(x, u, "quotient")


class WorldFeatures(nn.Module):
    """World ``x, u`` → 16-D ``(log R, p, v, ω, Ω)``."""

    dim = 16

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return features_from_state(x, u, "world")


class Orbit(nn.Module):
    """Reynolds average of a 13-D → 12-D body residual."""

    def __init__(self, residual: nn.Module) -> None:
        super().__init__()
        self.residual = residual

    def forward(self, canonical: torch.Tensor) -> torch.Tensor:
        batch = canonical.shape[0]
        stacked = orbit_features(canonical).reshape(4 * batch, 13)
        raw = self.residual(stacked).reshape(4, batch, 12)
        return orbit_body_dx(raw)
