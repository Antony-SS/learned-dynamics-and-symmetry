"""SO(3) helpers for a raw body-frame quadrotor step."""

from __future__ import annotations

import numpy as np

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
    """ZYX Euler angles in radians, matching the NanoBench quaternion."""
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
