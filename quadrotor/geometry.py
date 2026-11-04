"""SO(3) helpers and the shared kinematics step.

Numpy maps live next to the batched torch versions so IDSIA and NeuroBEM
do not depend on the leftover GRU trainer.
"""

from __future__ import annotations

import numpy as np
import torch

from quadrotor.canonical import PreparedFlight

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


def so3_log_batch(rotations: np.ndarray) -> np.ndarray:
    """Rotation vectors of a stack of matrices. Shape ``(N, 3, 3)`` to ``(N, 3)``."""
    rotations = np.asarray(rotations, dtype=float)
    cosine = np.clip((rotations[:, 0, 0] + rotations[:, 1, 1] + rotations[:, 2, 2] - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cosine)
    vee = np.stack(
        (
            rotations[:, 2, 1] - rotations[:, 1, 2],
            rotations[:, 0, 2] - rotations[:, 2, 0],
            rotations[:, 1, 0] - rotations[:, 0, 1],
        ),
        axis=1,
    )
    scale = np.empty_like(theta)
    small = theta <= _ANGLE_EPS
    scale[small] = 0.5
    scale[~small] = theta[~small] / (2.0 * np.sin(theta[~small]))
    return scale[:, None] * vee


def quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Scalar-last quaternions ``(N, 4)`` to rotation matrices ``(N, 3, 3)``."""
    quat = np.asarray(quat_xyzw, dtype=float)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.clip(norm, 1e-12, None)
    x, y, z, w = quat.T
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = np.empty((len(quat), 3, 3), dtype=float)
    matrix[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[:, 0, 1] = 2.0 * (xy - wz)
    matrix[:, 0, 2] = 2.0 * (xz + wy)
    matrix[:, 1, 0] = 2.0 * (xy + wz)
    matrix[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[:, 1, 2] = 2.0 * (yz - wx)
    matrix[:, 2, 0] = 2.0 * (xz - wy)
    matrix[:, 2, 1] = 2.0 * (yz + wx)
    matrix[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrix


def roll_pitch_yaw(rotations: np.ndarray) -> np.ndarray:
    """ZYX Euler angles in radians from rotation matrices."""
    rotations = np.asarray(rotations, dtype=float)
    pitch = np.arctan2(
        -rotations[:, 2, 0],
        np.hypot(rotations[:, 2, 1], rotations[:, 2, 2]),
    )
    roll = np.arctan2(rotations[:, 2, 1], rotations[:, 2, 2])
    yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
    return np.stack((roll, pitch, yaw), axis=1)


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


def train_device(requested: str | torch.device = "cpu") -> torch.device:
    requested = torch.device(requested)
    if requested.type == "cpu" and torch.backends.mps.is_available():
        return torch.device("mps")
    return requested


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
    scale = torch.where(sine < _ANGLE_EPS, torch.ones_like(sine), theta / sine)
    return scale.unsqueeze(-1) * half_vee


def _features(
    model: torch.nn.Module,
    position: torch.Tensor,
    rotation: torch.Tensor,
    velocity_world: torch.Tensor,
    omega_body: torch.Tensor,
    motors: torch.Tensor,
) -> torch.Tensor:
    if getattr(model, "reads_pose", False):
        features = torch.cat((position, velocity_world, _so3_log(rotation), omega_body, motors), dim=-1)
        if getattr(model, "reads_height", False):
            features = torch.cat((features, position[..., 2:3]), dim=-1)
        return features
    velocity_body = torch.bmm(rotation.transpose(1, 2), velocity_world.unsqueeze(-1)).squeeze(-1)
    up_world = rotation.new_tensor((0.0, 0.0, 1.0)).expand(rotation.shape[0], 3)
    gravity_body = torch.bmm(rotation.transpose(1, 2), up_world.unsqueeze(-1)).squeeze(-1)
    features = torch.cat((gravity_body, velocity_body, omega_body, motors), dim=-1)
    if getattr(model, "reads_height", False):
        features = torch.cat((features, position[..., 2:3]), dim=-1)
    return features


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


def window_starts(flights: list[PreparedFlight], horizon: int, stride: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for index, flight in enumerate(flights):
        last = flight.n_steps - horizon
        if last < 0:
            raise ValueError(
                f"{flight.name} has {flight.n_steps} steps, shorter than a {horizon} step window"
            )
        for start in range(0, last + 1, stride):
            pairs.append((index, start))
    if not pairs:
        raise ValueError("no training windows")
    return pairs


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
                model,
                position,
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


def prime_hidden(
    model: torch.nn.Module,
    flights: list[PreparedFlight],
    pairs: list[tuple[int, int]],
    warm_steps: int,
) -> torch.Tensor | None:
    """Teacher-force ``warm_steps`` Vicon frames ending at each window seed."""
    if warm_steps <= 0:
        return None
    device = module_device(model)
    reference = torch.zeros(len(pairs), 1, device=device)
    hidden = model.initial_hidden(len(pairs), reference)
    if hidden.numel() == 0:
        return hidden
    for offset in range(warm_steps):
        position, rotation, velocity, omega, motors = [], [], [], [], []
        for index, start in pairs:
            flight = flights[index]
            step = min(max(0, start - warm_steps + offset), flight.n_steps - 1)
            position.append(flight.position[step])
            rotation.append(flight.rotation[step])
            velocity.append(flight.velocity_world[step])
            omega.append(flight.omega_body[step])
            motors.append(flight.motors[step])
        features = _features(
            model,
            torch.as_tensor(np.stack(position), device=device),
            torch.as_tensor(np.stack(rotation), device=device),
            torch.as_tensor(np.stack(velocity), device=device),
            torch.as_tensor(np.stack(omega), device=device),
            torch.as_tensor(np.stack(motors), device=device),
        )
        step_fn = getattr(model, "step", None)
        if callable(step_fn):
            _acceleration, hidden = step_fn(features, hidden)
        else:
            neural = getattr(model, "neural", None)
            if neural is not None and callable(getattr(neural, "step", None)):
                _acceleration, hidden = neural.step(features, hidden)
    return hidden
