"""SO(3) helpers and the shared kinematics step."""

from __future__ import annotations

import numpy as np
import torch

_ANGLE_EPS = 1e-8


def hat(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rodrigues map from a rotation vector to a rotation matrix."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    skew = hat(phi)
    if theta <= _ANGLE_EPS:
        return np.eye(3) + skew
    skew_unit = skew / theta
    return np.eye(3) + np.sin(theta) * skew_unit + (1.0 - np.cos(theta)) * (skew_unit @ skew_unit)


# SO(3) log branches. θ = atan2(sin θ, cos θ) is accurate at every angle; only the
# axis needs care. Near 0 the scale θ / sin θ uses its Taylor series. Near π the
# skew part of R vanishes, so the axis comes from the symmetric part instead:
# (R + Rᵀ)/2 - cos θ I = (1 - cos θ) a aᵀ, with the sign taken from the skew part.
_LOG_SMALL_SINE = 1e-4
_LOG_NEAR_PI_SINE = 0.1  # with cos θ < 0, i.e. θ > ~174 deg


def so3_log_batch(rotations: np.ndarray) -> np.ndarray:
    """Rotation vectors of a stack of matrices. Shape ``(N, 3, 3)`` to ``(N, 3)``."""
    rotations = np.asarray(rotations, dtype=float)
    cosine = np.clip((np.trace(rotations, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    half_vee = 0.5 * np.stack(
        (
            rotations[:, 2, 1] - rotations[:, 1, 2],
            rotations[:, 0, 2] - rotations[:, 2, 0],
            rotations[:, 1, 0] - rotations[:, 0, 1],
        ),
        axis=1,
    )
    sine = np.linalg.norm(half_vee, axis=1)
    theta = np.arctan2(sine, cosine)
    small = (sine < _LOG_SMALL_SINE) & (cosine > 0.0)
    near_pi = (sine < _LOG_NEAR_PI_SINE) & (cosine < 0.0)
    out = np.empty_like(half_vee)
    out[small] = (1.0 + theta[small] ** 2 / 6.0)[:, None] * half_vee[small]
    generic = ~(small | near_pi)
    out[generic] = (theta[generic] / sine[generic])[:, None] * half_vee[generic]
    if near_pi.any():
        r = rotations[near_pi]
        c = cosine[near_pi]
        sym = 0.5 * (r + np.swapaxes(r, 1, 2)) - c[:, None, None] * np.eye(3)
        diag = np.diagonal(sym, axis1=1, axis2=2)
        k = np.argmax(diag, axis=1)
        rows = np.arange(len(k))
        axis = sym[rows, :, k] / np.sqrt(np.maximum(diag[rows, k] * (1.0 - c), 1e-300))[:, None]
        sign = np.where(np.einsum("ni,ni->n", axis, half_vee[near_pi]) < 0.0, -1.0, 1.0)
        out[near_pi] = (sign * theta[near_pi])[:, None] * axis
    return out


def project_rotation(rotation: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix, so an integrated attitude stays in SO(3)."""
    u, _singular, vt = np.linalg.svd(rotation)
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def module_device(model: torch.nn.Module) -> torch.device:
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


def train_device(requested: str | torch.device | None = None) -> torch.device:
    """The requested device. ``None`` or ``"auto"`` picks MPS when available, else CPU."""
    if requested is None or requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(requested)


def _hat(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zeros = torch.zeros_like(x)
    row0 = torch.stack((zeros, -z, y), dim=-1)
    row1 = torch.stack((z, zeros, -x), dim=-1)
    row2 = torch.stack((-y, x, zeros), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _so3_exp(phi: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True)
    small = theta < _ANGLE_EPS
    theta_safe = theta.clamp_min(_ANGLE_EPS)
    axis = phi / theta_safe
    skew = _hat(axis)
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(phi.shape[0], 3, 3)
    sine = torch.sin(theta_safe)[..., None]
    cosine = torch.cos(theta_safe)[..., None]
    rotation = eye + sine * skew + (1.0 - cosine) * (skew @ skew)
    return torch.where(small[..., None], eye + _hat(phi), rotation)


def _so3_log(rotation: torch.Tensor) -> torch.Tensor:
    cosine = ((torch.diagonal(rotation, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    half_vee = 0.5 * torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.linalg.vector_norm(half_vee, dim=-1)
    theta = torch.atan2(sine, cosine)
    small = (sine < _LOG_SMALL_SINE) & (cosine > 0.0)
    near_pi = (sine < _LOG_NEAR_PI_SINE) & (cosine < 0.0)
    # Every branch is evaluated, so denominators are made safe on the rows a
    # branch does not own; otherwise their NaN/inf would leak into the gradient.
    sine_safe = torch.where(small | near_pi, torch.ones_like(sine), sine)
    scale = torch.where(small, 1.0 + theta.square() / 6.0, theta / sine_safe)
    generic = scale.unsqueeze(-1) * half_vee
    if not bool(near_pi.any()):
        return generic
    eye = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    sym = 0.5 * (rotation + rotation.transpose(-1, -2)) - cosine[..., None, None] * eye
    diag = torch.diagonal(sym, dim1=-2, dim2=-1)
    k = diag.argmax(dim=-1, keepdim=True)
    column = torch.gather(sym, -1, k.unsqueeze(-2).expand(*sym.shape[:-1], 1)).squeeze(-1)
    diag_k = torch.gather(diag, -1, k).squeeze(-1)
    denom = torch.where(near_pi, diag_k * (1.0 - cosine), torch.ones_like(diag_k)).clamp_min(1e-30).sqrt()
    axis = column / denom.unsqueeze(-1)
    sign = torch.where((axis * half_vee).sum(-1) < 0.0, -torch.ones_like(sine), torch.ones_like(sine))
    return torch.where(near_pi.unsqueeze(-1), (sign * theta).unsqueeze(-1) * axis, generic)


def _features(
    rotation: torch.Tensor,
    velocity_world: torch.Tensor,
    omega_body: torch.Tensor,
    motors: torch.Tensor,
) -> torch.Tensor:
    velocity_body = torch.bmm(rotation.transpose(1, 2), velocity_world.unsqueeze(-1)).squeeze(-1)
    up_world = rotation.new_tensor((0.0, 0.0, 1.0)).expand(rotation.shape[0], 3)
    gravity_body = torch.bmm(rotation.transpose(1, 2), up_world.unsqueeze(-1)).squeeze(-1)
    return torch.cat((gravity_body, velocity_body, omega_body, motors), dim=-1)


def _apply(
    position: torch.Tensor,
    rotation: torch.Tensor,
    velocity_world: torch.Tensor,
    omega_body: torch.Tensor,
    acceleration: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dt = dt.unsqueeze(-1)
    acceleration_world = torch.bmm(rotation, acceleration[:, 0:3].unsqueeze(-1)).squeeze(-1)
    angular_acceleration = acceleration[:, 3:6]
    position = position + velocity_world * dt + 0.5 * acceleration_world * dt.square()
    velocity_world = velocity_world + acceleration_world * dt
    rotation_increment = (omega_body + 0.5 * angular_acceleration * dt) * dt
    rotation = torch.bmm(rotation, _so3_exp(rotation_increment))
    omega_body = omega_body + angular_acceleration * dt
    return position, rotation, velocity_world, omega_body


def _integrate(
    model: torch.nn.Module,
    motors: torch.Tensor,
    dt: torch.Tensor,
    position: torch.Tensor,
    rotation: torch.Tensor,
    velocity_world: torch.Tensor,
    omega_body: torch.Tensor,
    hidden: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions, rotations, velocities, omegas = [], [], [], []
    if hidden is None:
        hidden = model.initial_hidden(motors.shape[0], motors)
    advance = getattr(model, "advance", None)
    for index in range(motors.shape[1]):
        if callable(advance):
            position, rotation, velocity_world, omega_body, hidden = advance(
                position,
                rotation,
                velocity_world,
                omega_body,
                motors[:, index],
                dt[:, index],
                hidden,
            )
        else:
            features = _features(
                rotation,
                velocity_world,
                omega_body,
                motors[:, index],
            )
            acceleration, hidden = model.step(features, hidden)
            position, rotation, velocity_world, omega_body = _apply(
                position, rotation, velocity_world, omega_body, acceleration, dt[:, index]
            )
        positions.append(position)
        rotations.append(rotation)
        velocities.append(velocity_world)
        omegas.append(omega_body)
    return (
        torch.stack(positions, dim=1),
        torch.stack(rotations, dim=1),
        torch.stack(velocities, dim=1),
        torch.stack(omegas, dim=1),
    )
