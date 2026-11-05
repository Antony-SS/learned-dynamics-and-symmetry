"""Quotient features, Z₂ orbit, and the body increment lift.

Tables live here once. Residual MLPs stay geometry-free: they write body
``(Δx, Δy, Δθ, Δv, Δω)``. Polar pose parts lift through the current heading;
rates add in the body.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Two Z₂ elements: identity, then left–right (reflect through the body xz plane).
# Polar (v, Δx, Δy, Δv): v, Δx, Δv stay; Δy flips.
# Axial / pseudo (ω, Δθ, Δω, ω_cmd): pick up det = −1.
# Wheel swap (left, right) → (right, left) is exactly “v stays, ω flips”.
BASE_LENGTH = 0.105
ORBIT_POLAR = (
    (1.0, 1.0),
    (1.0, -1.0),
)
ORBIT_AXIAL = (
    (1.0,),
    (-1.0,),
)
N_ORBIT = 2
# Quotient row (v, ω, v_cmd, ω_cmd) and body leftover (Δx, Δy, Δθ, Δv, Δω).
ORBIT_FEAT_SIGN = (
    (1.0, 1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0, -1.0),
)
ORBIT_DX_SIGN = (
    (1.0, 1.0, 1.0, 1.0, 1.0),
    (1.0, -1.0, -1.0, 1.0, -1.0),
)


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def twist_to_wheels(vw: torch.Tensor) -> torch.Tensor:
    """``(v, ω)`` → ``(v_left, v_right)`` at ``BASE_LENGTH``."""
    half = 0.5 * BASE_LENGTH
    return torch.stack((vw[..., 0] - vw[..., 1] * half, vw[..., 0] + vw[..., 1] * half), dim=-1)


def wheels_to_twist(wheels: torch.Tensor) -> torch.Tensor:
    """``(v_left, v_right)`` → ``(v, ω)``."""
    return torch.stack(
        (0.5 * (wheels[..., 0] + wheels[..., 1]), (wheels[..., 1] - wheels[..., 0]) / BASE_LENGTH),
        dim=-1,
    )


def apply_pose_dx(pose: torch.Tensor, dx_pose: torch.Tensor) -> torch.Tensor:
    """``T ∘ Exp(Δ)`` with ``dx_pose = (Δx, Δy, Δθ)`` already a finite step."""
    cosine = torch.cos(pose[..., 2])
    sine = torch.sin(pose[..., 2])
    x = pose[..., 0] + cosine * dx_pose[..., 0] - sine * dx_pose[..., 1]
    y = pose[..., 1] + sine * dx_pose[..., 0] + cosine * dx_pose[..., 1]
    theta = wrap_angle(pose[..., 2] + dx_pose[..., 2])
    return torch.stack((x, y, theta), dim=-1)


def apply_body_dx(
    pose: torch.Tensor, rates: torch.Tensor, dx_body: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add a 5-D body increment. Polar (Δx, Δy) lift through heading; rates add."""
    next_pose = apply_pose_dx(pose, dx_body[..., 0:3])
    next_rates = rates + dx_body[..., 3:5]
    return next_pose, next_rates


def physics_body_dx(command: torch.Tensor, dt: float) -> torch.Tensor:
    """Ideal unicycle step from commanded ``(v, ω)``."""
    zeros = torch.zeros_like(command[..., 0])
    return torch.stack((command[..., 0] * dt, zeros, command[..., 1] * dt), dim=-1)


def world_features(pose: torch.Tensor, rates: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    """7-D ``(x, y, θ, v, ω, v_cmd, ω_cmd)``."""
    return torch.cat((pose, rates, command), dim=-1)


def quotient_features(rates: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    """4-D ``(v, ω, v_cmd, ω_cmd)``. Drops arena pose."""
    return torch.cat((rates, command), dim=-1)


def orbit_features(canonical: torch.Tensor) -> torch.Tensor:
    """Stack identity and the left–right image. Shape ``(2, B, 4)``."""
    signs = canonical.new_tensor(ORBIT_FEAT_SIGN)
    return canonical.unsqueeze(0) * signs.unsqueeze(1)


def apply_orbit_element(canonical: torch.Tensor, index: int) -> torch.Tensor:
    """Apply one Z₂ element to a 4-D quotient row."""
    return orbit_features(canonical)[index]


def orbit_body_dx(raw: torch.Tensor) -> torch.Tensor:
    """Sign-correct a ``(2, B, 5)`` body increment and Reynolds-average."""
    signs = raw.new_tensor(ORBIT_DX_SIGN)
    return (raw * signs.unsqueeze(1)).mean(dim=0)


def mirror_quotient(features: torch.Tensor) -> torch.Tensor:
    return apply_orbit_element(features, 1)


def mirror_body_dx(dx_body: torch.Tensor) -> torch.Tensor:
    return dx_body * dx_body.new_tensor(ORBIT_DX_SIGN[1])


class Quotient(nn.Module):
    dim = 4

    def forward(self, pose: torch.Tensor, rates: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        del pose
        return quotient_features(rates, command)


class WorldFeatures(nn.Module):
    dim = 7

    def forward(self, pose: torch.Tensor, rates: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        return world_features(pose, rates, command)


class Orbit(nn.Module):
    """Reynolds average: one MLP on identity and the left–right image, then mean."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("feat_sign", torch.tensor(ORBIT_FEAT_SIGN))
        self.register_buffer("dx_sign", torch.tensor(ORBIT_DX_SIGN))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        stacked = (features.unsqueeze(0) * self.feat_sign.unsqueeze(1)).reshape(N_ORBIT * batch, 4)
        raw = self.net(stacked).reshape(N_ORBIT, batch, 5)
        return (raw * self.dx_sign.unsqueeze(1)).mean(dim=0)
