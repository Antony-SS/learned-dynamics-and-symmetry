"""Quotient features, the left–right mirror average, and the body-increment lift.

This module is the reusable layer. It wraps *any* MLP ``ℝ⁴ → ℝ⁵`` so that the
resulting unicycle model is

* **invariant** to translating and rotating the arena, because the MLP only
  sees the 4-D quotient ``(v, ω, v_cmd, ω_cmd)``; and
* **equivariant** to the left–right mirror (swapping the wheels), because
  :class:`SymmetricQuotientMLP` averages the MLP over both mirror images.

The MLP itself is unconstrained. It writes a body-frame increment
``(Δx, Δy, Δθ, Δv, Δω)``, including sideslip Δy (no no-slip constraint is
imposed), and :func:`apply_body_dx` lifts the increment to the next pose and rates.

Typical use::

    net = MyMLP(in_dim=4, out_dim=5)            # any nn.Module
    model = SymmetricQuotientMLP(net)
    mean, scale = symmetrize_feature_stats(mean, scale)   # see below
    dx = model((quotient_features(rates, command) - mean) / scale)
    next_pose, next_rates = apply_body_dx(pose, rates, dx)

Normalization contract
----------------------
``SymmetricQuotientMLP`` applies the mirror to its *input as given*. If you
standardize the features first, the standardization must commute with the
mirror, or the model is only approximately equivariant. That requires a zero
mean on ω and ω_cmd. Pass per-channel training statistics through
:func:`symmetrize_feature_stats`. It returns the statistics of the training
data together with its mirror image.
"""

from __future__ import annotations

import torch
import torch.nn as nn

BASE_LENGTH = 0.105
N_GROUP = 2
QUOTIENT_DIM = 4
INCREMENT_DIM = 5

# The two elements of Z₂: identity, then the left–right mirror (reflection
# through the body xz plane, which swaps the wheels).
#
# Polar quantities (v, v_cmd, Δx, Δv) are unchanged; sideslip Δy flips (it points
# along the mirrored body y axis). Axial quantities
# (ω, ω_cmd, Δθ, Δω) flip sign. Swapping the wheels, (left, right) ↦
# (right, left), is exactly "v stays, ω flips".
#
# Quotient row: (v, ω, v_cmd, ω_cmd).
GROUP_FEAT_SIGN = (
    (1.0, 1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0, -1.0),
)
# Increment: (Δx, Δy, Δθ, Δv, Δω).
GROUP_DX_SIGN = (
    (1.0, 1.0, 1.0, 1.0, 1.0),
    (1.0, -1.0, -1.0, 1.0, -1.0),
)


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``[-π, π)``."""
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def twist_to_wheels(vw: torch.Tensor) -> torch.Tensor:
    """``(v, ω)`` → ``(v_left, v_right)`` for a wheel base of ``BASE_LENGTH``."""
    half = 0.5 * BASE_LENGTH
    return torch.stack((vw[..., 0] - vw[..., 1] * half, vw[..., 0] + vw[..., 1] * half), dim=-1)


def wheels_to_twist(wheels: torch.Tensor) -> torch.Tensor:
    """``(v_left, v_right)`` → ``(v, ω)``."""
    return torch.stack(
        (0.5 * (wheels[..., 0] + wheels[..., 1]), (wheels[..., 1] - wheels[..., 0]) / BASE_LENGTH),
        dim=-1,
    )


def apply_pose_dx(pose: torch.Tensor, dx_pose: torch.Tensor) -> torch.Tensor:
    """Compose a pose with a finite body step ``dx_pose = (Δx, Δy, Δθ)``.

    The body translation is rotated by the current heading, and the heading is
    wrapped.
    """
    cosine = torch.cos(pose[..., 2])
    sine = torch.sin(pose[..., 2])
    x = pose[..., 0] + cosine * dx_pose[..., 0] - sine * dx_pose[..., 1]
    y = pose[..., 1] + sine * dx_pose[..., 0] + cosine * dx_pose[..., 1]
    theta = wrap_angle(pose[..., 2] + dx_pose[..., 2])
    return torch.stack((x, y, theta), dim=-1)


def apply_body_dx(
    pose: torch.Tensor, rates: torch.Tensor, dx_body: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a body increment ``(Δx, Δy, Δθ, Δv, Δω)``.

    Returns the next pose (body step composed onto ``pose``) and the next rates
    ``rates + (Δv, Δω)``.
    """
    next_pose = apply_pose_dx(pose, dx_body[..., 0:3])
    next_rates = rates + dx_body[..., 3:5]
    return next_pose, next_rates


def physics_body_dx(command: torch.Tensor, dt: float) -> torch.Tensor:
    """Ideal unicycle body step ``(v_cmd dt, 0, ω_cmd dt)``."""
    zeros = torch.zeros_like(command[..., 0])
    return torch.stack((command[..., 0] * dt, zeros, command[..., 1] * dt), dim=-1)


def world_features(pose: torch.Tensor, rates: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    """7-D world features ``(x, y, θ, v, ω, v_cmd, ω_cmd)``. No symmetry is removed."""
    return torch.cat((pose, rates, command), dim=-1)


def quotient_features(rates: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    """4-D quotient features ``(v, ω, v_cmd, ω_cmd)``. The arena pose is dropped."""
    return torch.cat((rates, command), dim=-1)


def group_features(features: torch.Tensor) -> torch.Tensor:
    """Both mirror images of 4-D quotient rows. Shape ``(2, B, 4)``."""
    signs = features.new_tensor(GROUP_FEAT_SIGN)
    return features.unsqueeze(0) * signs.unsqueeze(1)


def apply_group_element(features: torch.Tensor, index: int) -> torch.Tensor:
    """Apply group element ``index`` (0 = identity, 1 = mirror) to 4-D quotient rows."""
    return group_features(features)[index]


def average_body_dx(raw: torch.Tensor) -> torch.Tensor:
    """Map both ``(2, B, 4)`` increments back by their mirror, then average.

    The mirror is its own inverse, so the same sign table maps forward and back.
    """
    signs = raw.new_tensor(GROUP_DX_SIGN)
    return (raw * signs.unsqueeze(1)).mean(dim=0)


def symmetrize_feature_stats(mean: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror-invariant mean and std for the 4-D quotient features.

    Given per-channel ``mean`` and ``scale`` (std) from the training data, this
    returns the statistics of that data together with its mirror image. ω and
    ω_cmd get mean 0 and std ``sqrt(std² + mean²)``. v and v_cmd are unchanged.

    Standardizing with these statistics commutes with the mirror, which is what
    :class:`SymmetricQuotientMLP` needs to be exactly equivariant.
    """
    second_moment = scale.square() + mean.square()
    sym_mean = group_features(mean.unsqueeze(0)).mean(dim=0).squeeze(0)
    # The mirror only flips signs, so squared channels are unchanged.
    sym_second = group_features(second_moment.unsqueeze(0)).abs().mean(dim=0).squeeze(0)
    sym_scale = (sym_second - sym_mean.square()).clamp_min(0.0).sqrt()
    return sym_mean, sym_scale


def mirror_quotient(features: torch.Tensor) -> torch.Tensor:
    """Left–right mirror of 4-D quotient rows."""
    return apply_group_element(features, 1)


def mirror_body_dx(dx_body: torch.Tensor) -> torch.Tensor:
    """Left–right mirror of an increment ``(Δx, Δθ, Δv, Δω)``."""
    return dx_body * dx_body.new_tensor(GROUP_DX_SIGN[1])


class Quotient(nn.Module):
    """``(pose, rates, command)`` → 4-D quotient features. The pose is ignored."""

    dim = QUOTIENT_DIM

    def forward(self, pose: torch.Tensor, rates: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        del pose
        return quotient_features(rates, command)


class WorldFeatures(nn.Module):
    """``(pose, rates, command)`` → 7-D world features."""

    dim = 7

    def forward(self, pose: torch.Tensor, rates: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        return world_features(pose, rates, command)


class SymmetricQuotientMLP(nn.Module):
    """Make any ``ℝ⁴ → ℝ⁴`` network exactly equivariant to the left–right mirror.

    For each input row ``z`` the wrapper computes::

        f_sym(z) = ½ [ net(z) + m · net(m · z) ]

    ``m`` flips ω and ω_cmd on the input, and Δθ and Δω on the output. This is
    the group average (Reynolds operator): the result is equivariant for any
    ``net``. A straight-line command (ω = ω_cmd = 0) therefore always predicts
    zero turning. Both images go through ``net`` as one batch.

    Args:
        net: Any module mapping ``(N, 4)`` to ``(N, 4)``. It is not constrained
            or modified.

    Input:
        ``(B, 4)`` quotient features ``(v, ω, v_cmd, ω_cmd)``. If you
        standardize them, use :func:`symmetrize_feature_stats` (see the module
        docstring).

    Output:
        ``(B, 5)`` body increment ``(Δx, Δy, Δθ, Δv, Δω)``.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("feat_sign", torch.tensor(GROUP_FEAT_SIGN))
        self.register_buffer("dx_sign", torch.tensor(GROUP_DX_SIGN))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        stacked = (features.unsqueeze(0) * self.feat_sign.unsqueeze(1)).reshape(N_GROUP * batch, QUOTIENT_DIM)
        raw = self.net(stacked).reshape(N_GROUP, batch, INCREMENT_DIM)
        return (raw * self.dx_sign.unsqueeze(1)).mean(dim=0)
