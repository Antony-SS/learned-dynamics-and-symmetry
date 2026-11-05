"""Quotient features, the Z₂×Z₂ mirror average, and the body-increment lift.

This module wraps *any* MLP ``ℝ¹³ → ℝ¹²`` so that
the resulting dynamics model is

* **invariant** to horizontal translation and yaw, because the MLP only sees the
  13-D quotient ``(g_body, v_body, ω, Ω)``; and
* **equivariant** to the airframe mirrors (body xz and yz planes), because
  :class:`Orbit` averages the MLP over all four mirror images.

The MLP itself is unconstrained. It writes a body-frame increment
``(Δp, Δv, Δθ, Δω)``, and :func:`apply_body_dx` lifts that increment back to the
world frame.

Typical use::

    net = MyMLP(in_dim=13, out_dim=12)          # any nn.Module
    model = Orbit(net)
    mean, scale = symmetrize_feature_stats(mean, scale)   # see below
    dx_body = model((quotient_features - mean) / scale)  # (B, 12)
    x_next = apply_body_dx(x_phys, x, dx_body)

Normalization contract
----------------------
``Orbit`` applies the mirrors to its *input as given*. If you standardize the
features first, the standardization must commute with the mirrors, or the model
is only approximately equivariant. That requires a zero mean on every
sign-flipped channel and the same mean and scale for all four motors. Pass
per-channel training statistics through :func:`symmetrize_feature_stats`. It
returns the statistics of the training data together with all of its mirror
images.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quadrotor.geometry import _features, _so3_exp, _so3_log

N_GROUP = 4
QUOTIENT_DIM = 13
INCREMENT_DIM = 12

# The four elements of Z₂ × Z₂, in order: identity, mirror through the body xz
# plane (y ↦ −y), mirror through the body yz plane (x ↦ −x), and both (a half
# turn about body z).
#
# Polar vectors (g, v, Δp, Δv) flip the mirrored axis. Axial vectors
# (ω, Δθ, Δω) pick up an extra factor det = −1 for a single mirror.
#
# Rotor speeds Ω are in Crazyflie-X order: m1 front-right, m2 back-right,
# m3 back-left, m4 front-left. A mirror swaps rotors so each one stays in the
# same geometric corner. Rotor chirality also flips, so a mirrored CW rotor is
# the CCW rotor it lands on.
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
ORBIT_MOTORS = (
    (0, 1, 2, 3),
    (3, 2, 1, 0),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
)
# Signs on the first six quotient channels (g_body, v_body): both are polar.
ORBIT_GV = tuple(polar + polar for polar in ORBIT_POLAR)
# Signs on the body rate ω (axial).
ORBIT_W = ORBIT_AXIAL


def quotient(
    rotation: torch.Tensor,
    velocity_world: torch.Tensor,
    omega: torch.Tensor,
    motors: torch.Tensor,
) -> torch.Tensor:
    """13-D quotient features ``(g_body, v_body, ω, Ω)``.

    ``g_body = Rᵀ e_z`` and ``v_body = Rᵀ v``. Position and yaw are dropped.
    """
    return _features(rotation, velocity_world, omega, motors)


def world_features(
    position: torch.Tensor,
    rotation: torch.Tensor,
    velocity_world: torch.Tensor,
    omega: torch.Tensor,
    motors: torch.Tensor,
) -> torch.Tensor:
    """16-D world features ``(log R, p, v, ω, Ω)``. No symmetry is removed."""
    return torch.cat((_so3_log(rotation), position, velocity_world, omega, motors), dim=-1)


def features_from_state(x: torch.Tensor, u: torch.Tensor, kind: str) -> torch.Tensor:
    """Features from a packed state ``x = (p, v, log R, ω)`` and rotor speeds ``u``."""
    pos, vel, so3, omega = x.split((3, 3, 3, 3), dim=-1)
    rotation = _so3_exp(so3)
    if kind == "world":
        return world_features(pos, rotation, vel, omega, u)
    if kind == "quotient":
        return quotient(rotation, vel, omega, u)
    raise ValueError(f"unknown features {kind}")


def apply_orbit_element(features: torch.Tensor, index: int) -> torch.Tensor:
    """Apply group element ``index`` (0–3) to 13-D quotient rows."""
    gv = features[..., 0:6] * features.new_tensor(ORBIT_GV[index])
    omega = features[..., 6:9] * features.new_tensor(ORBIT_W[index])
    motors = features[..., 9:13][..., list(ORBIT_MOTORS[index])]
    return torch.cat((gv, omega, motors), dim=-1)


def orbit_features(features: torch.Tensor) -> torch.Tensor:
    """All four mirror images of ``(B, 13)`` quotient rows. Shape ``(4, B, 13)``."""
    batch = features.shape[0]
    gv = features.new_tensor(ORBIT_GV)
    omega = features.new_tensor(ORBIT_W)
    motor_index = torch.tensor(ORBIT_MOTORS, device=features.device, dtype=torch.long)
    motor_index = motor_index.view(N_GROUP, 1, 4).expand(N_GROUP, batch, 4)
    motors = features[:, 9:13].unsqueeze(0).expand(N_GROUP, -1, -1).gather(-1, motor_index)
    return torch.cat(
        (
            features[:, 0:6].unsqueeze(0) * gv.view(N_GROUP, 1, 6),
            features[:, 6:9].unsqueeze(0) * omega.view(N_GROUP, 1, 3),
            motors,
        ),
        dim=-1,
    )


def orbit_body_dx(raw: torch.Tensor) -> torch.Tensor:
    """Map each of four ``(4, B, 12)`` increments back by its mirror, then average.

    Every element of Z₂ × Z₂ is its own inverse, so the same sign table maps an
    increment forward and back.
    """
    polar = raw.new_tensor(ORBIT_POLAR).view(N_GROUP, 1, 3)
    axial = raw.new_tensor(ORBIT_AXIAL).view(N_GROUP, 1, 3)
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


def symmetrize_feature_stats(mean: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror-invariant mean and std for the 13-D quotient features.

    Given per-channel ``mean`` and ``scale`` (std) from the training data, this
    returns the statistics of that data together with its four mirror images.
    Every sign-flipped channel gets mean 0. The four motors share one mean and
    one std. ``g_z`` and ``v_z`` are unchanged.

    Standardizing with these statistics commutes with every mirror, which is
    what :class:`Orbit` needs to be exactly equivariant.
    """
    second_moment = scale.square() + mean.square()
    sym_mean = orbit_features(mean.unsqueeze(0)).mean(dim=0).squeeze(0)
    # Each mirror is a signed permutation, so it permutes squared channels.
    sym_second = orbit_features(second_moment.unsqueeze(0)).abs().mean(dim=0).squeeze(0)
    sym_scale = (sym_second - sym_mean.square()).clamp_min(0.0).sqrt()
    return sym_mean, sym_scale


def apply_body_dx(x_phys: torch.Tensor, x: torch.Tensor, dx_body: torch.Tensor) -> torch.Tensor:
    """Add a body increment to a base step and return the packed next state.

    ``x`` is the current state ``(p, v, log R, ω)``. ``x_phys`` is the base step:
    the RK4 physics prediction, or ``x`` itself when there is no physics model.
    The polar parts are lifted through the *current* attitude ``R``, and the
    rotation composes on SO(3)::

        p' = p_phys + R Δp
        v' = v_phys + R Δv
        R' = R_phys Exp(Δθ)
        ω' = ω_phys + Δω
    """
    delta_pos, delta_vel, delta_rot, delta_omega = dx_body.split((3, 3, 3, 3), dim=-1)
    pos_p, vel_p, so3_p, omega_p = x_phys.split((3, 3, 3, 3), dim=-1)
    rotation = _so3_exp(x[:, 6:9])
    pos_next = pos_p + torch.bmm(rotation, delta_pos.unsqueeze(-1)).squeeze(-1)
    vel_next = vel_p + torch.bmm(rotation, delta_vel.unsqueeze(-1)).squeeze(-1)
    rotation_next = torch.bmm(_so3_exp(so3_p), _so3_exp(delta_rot))
    omega_next = omega_p + delta_omega
    return torch.cat((pos_next, vel_next, _so3_log(rotation_next), omega_next), dim=-1)


class Quotient(nn.Module):
    """Packed state ``x`` and rotor speeds ``u`` → 13-D quotient features."""

    dim = QUOTIENT_DIM

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return features_from_state(x, u, "quotient")


class WorldFeatures(nn.Module):
    """Packed state ``x`` and rotor speeds ``u`` → 16-D ``(log R, p, v, ω, Ω)``."""

    dim = 16

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return features_from_state(x, u, "world")


class Orbit(nn.Module):
    """Make any ``ℝ¹³ → ℝ¹²`` network exactly equivariant to the Z₂×Z₂ airframe mirrors.

    For each input row ``z`` the wrapper computes::

        f_sym(z) = ¼ Σ_g  g⁻¹ · residual(g · z)

    ``g`` runs over the four mirrors. On the input it flips the polar and axial
    channels and swaps rotors (:func:`apply_orbit_element`). On the output it
    applies the matching polar and axial signs (:func:`orbit_body_dx`). This is
    the group average (Reynolds operator): the result is equivariant for any
    ``residual``. All four images go through ``residual`` as one batch.

    Args:
        residual: Any module mapping ``(N, 13)`` to ``(N, 12)``. It is not
            constrained or modified.

    Input:
        ``(B, 13)`` quotient features ``(g_body, v_body, ω, Ω)``. If you
        standardize them, use :func:`symmetrize_feature_stats` (see the module
        docstring).

    Output:
        ``(B, 12)`` body increment ``(Δp, Δv, Δθ, Δω)``.
    """

    def __init__(self, residual: nn.Module) -> None:
        super().__init__()
        self.residual = residual

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        stacked = orbit_features(features).reshape(N_GROUP * batch, QUOTIENT_DIM)
        raw = self.residual(stacked).reshape(N_GROUP, batch, INCREMENT_DIM)
        return orbit_body_dx(raw)
