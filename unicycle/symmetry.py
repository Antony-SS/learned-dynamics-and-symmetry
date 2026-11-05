"""Quotient features, Z₂ orbit, and the body increment lift.

Tables live here once. Residual MLPs stay geometry-free: they write body
``(Δx, Δy, Δθ)``. Polar parts are lifted by the current heading.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Two Z₂ elements: identity, then left–right (reflect through the body xz plane).
# Polar (v, Δx, Δy): v and Δx stay; Δy flips.
# Axial / pseudo (ω, Δθ, ω_cmd): pick up det = −1.
# Wheel slots stay (left, right); the second row swaps them.
BASE_LENGTH = 0.105
ORBIT_VW = (
    (1.0, 1.0),
    (1.0, -1.0),
)
ORBIT_WHEELS = ((0, 1), (1, 0))
ORBIT_POLAR = (
    (1.0, 1.0),
    (1.0, -1.0),
)
ORBIT_AXIAL = (
    (1.0,),
    (-1.0,),
)
N_ORBIT = 2


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def _as_orbit_tensor(table, device, dtype, long: bool = False) -> torch.Tensor:
    return torch.tensor(table, device=device, dtype=torch.long if long else dtype)


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


def apply_body_dx(pose: torch.Tensor, dx_body: torch.Tensor) -> torch.Tensor:
    """``T ∘ Exp(Δ)`` with ``dx_body = (Δx, Δy, Δθ)`` already a finite step."""
    cosine = torch.cos(pose[..., 2])
    sine = torch.sin(pose[..., 2])
    x = pose[..., 0] + cosine * dx_body[..., 0] - sine * dx_body[..., 1]
    y = pose[..., 1] + sine * dx_body[..., 0] + cosine * dx_body[..., 1]
    theta = wrap_angle(pose[..., 2] + dx_body[..., 2])
    return torch.stack((x, y, theta), dim=-1)


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


def _permute_twist(vw: torch.Tensor, wheels_idx: torch.Tensor) -> torch.Tensor:
    """Apply a wheel permutation to a ``(..., 2)`` twist, batched over the group."""
    wheels = twist_to_wheels(vw)
    index = wheels_idx.view(N_ORBIT, *([1] * (wheels.ndim - 1)), 2).expand(N_ORBIT, *wheels.shape)
    return wheels_to_twist(wheels.unsqueeze(0).expand(N_ORBIT, *wheels.shape).gather(-1, index))


def orbit_features(canonical: torch.Tensor) -> torch.Tensor:
    """Stack identity and the left–right image. Shape ``(2, B, 4)``."""
    device, dtype = canonical.device, canonical.dtype
    wheels_idx = _as_orbit_tensor(ORBIT_WHEELS, device, dtype, long=True)
    rates = _permute_twist(canonical[..., 0:2], wheels_idx)
    command = _permute_twist(canonical[..., 2:4], wheels_idx)
    return torch.cat((rates, command), dim=-1)


def apply_orbit_element(canonical: torch.Tensor, index: int) -> torch.Tensor:
    """Apply one Z₂ element to a 4-D quotient row."""
    return orbit_features(canonical)[index]


def orbit_body_dx(raw: torch.Tensor) -> torch.Tensor:
    """Sign-correct a ``(2, B, 3)`` body increment and Reynolds-average."""
    polar = raw.new_tensor(ORBIT_POLAR).view(N_ORBIT, 1, 2)
    axial = raw.new_tensor(ORBIT_AXIAL).view(N_ORBIT, 1, 1)
    signed = torch.cat((raw[..., 0:2] * polar, raw[..., 2:3] * axial), dim=-1)
    return signed.mean(dim=0)


def mirror_quotient(features: torch.Tensor) -> torch.Tensor:
    return apply_orbit_element(features, 1)


def mirror_body_dx(dx_body: torch.Tensor) -> torch.Tensor:
    signs = dx_body.new_tensor((*ORBIT_POLAR[1], *ORBIT_AXIAL[1]))
    return dx_body * signs


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
    """Reynolds average of a 4-D → 3-D body increment over the two mirrors."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        stacked = orbit_features(features).reshape(N_ORBIT * batch, 4)
        raw = self.net(stacked).reshape(N_ORBIT, batch, 3)
        return orbit_body_dx(raw)
