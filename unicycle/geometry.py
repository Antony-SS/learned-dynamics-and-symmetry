"""SE(2) helpers for a raw pose step.

``Log`` and ``Exp`` use the GTSAM Pose2 chart: the tangent vector is
``(ρx, ρy, φ)`` and the relative pose translation is ``V(φ) ρ``. Integrating
``Exp(Log(T_t^{-1} T_{t+1}))`` recovers the next Vicon pose. That tangent
vector is the learning target. It is not a correction applied on top of a
unicycle step.
"""

from __future__ import annotations

import numpy as np

_ANGLE_EPS = 1e-8


def wrap_angle(angle):
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _v_coefficients(phi):
    """``(sin(φ)/φ, (1-cos(φ))/φ)``, with the small-angle values ``(1, φ/2)``."""
    phi = np.asarray(phi, dtype=float)
    a = np.empty(phi.shape, dtype=float)
    b = np.empty(phi.shape, dtype=float)
    small = np.abs(phi) <= _ANGLE_EPS
    a[small] = 1.0
    b[small] = 0.5 * phi[small]
    large = ~small
    ph = phi[large]
    a[large] = np.sin(ph) / ph
    b[large] = (1.0 - np.cos(ph)) / ph
    return a, b


def se2_log(translation_xy, phi):
    """Tangent coordinates of a relative pose whose translation is ``(x, y)``."""
    translation = np.asarray(translation_xy, dtype=float)
    single = translation.ndim == 1
    if single:
        translation = translation.reshape(1, 2)
        phi = np.asarray([phi], dtype=float)
    else:
        phi = np.asarray(phi, dtype=float).reshape(-1)
    x = translation[:, 0]
    y = translation[:, 1]
    a, b = _v_coefficients(phi)
    det = a * a + b * b
    rho_x = (a * x + b * y) / det
    rho_y = (-b * x + a * y) / det
    out = np.stack((rho_x, rho_y, phi), axis=1)
    return out[0] if single else out


def se2_exp(xi):
    """Relative pose ``(x, y, φ)`` from tangent coordinates ``(ρx, ρy, φ)``."""
    xi = np.asarray(xi, dtype=float)
    single = xi.ndim == 1
    if single:
        xi = xi.reshape(1, 3)
    phi = xi[:, 2]
    a, b = _v_coefficients(phi)
    x = a * xi[:, 0] - b * xi[:, 1]
    y = b * xi[:, 0] + a * xi[:, 1]
    out = np.stack((x, y, phi), axis=1)
    return out[0] if single else out


def body_steps(poses: np.ndarray) -> np.ndarray:
    """``Log(T_t^{-1} T_{t+1})`` for each consecutive pair. Shape ``(N-1, 3)``."""
    poses = np.asarray(poses, dtype=float)
    start = poses[:-1]
    stop = poses[1:]
    phi = wrap_angle(stop[:, 2] - start[:, 2])
    cosine = np.cos(start[:, 2])
    sine = np.sin(start[:, 2])
    dx = stop[:, 0] - start[:, 0]
    dy = stop[:, 1] - start[:, 1]
    body_x = cosine * dx + sine * dy
    body_y = -sine * dx + cosine * dy
    return se2_log(np.stack((body_x, body_y), axis=1), phi)


def relative_poses(poses: np.ndarray) -> np.ndarray:
    """Body-frame pose of ``T_t^{-1} T_{t+1}``, before the logarithm. Shape ``(N-1, 3)``."""
    poses = np.asarray(poses, dtype=float)
    start = poses[:-1]
    stop = poses[1:]
    phi = wrap_angle(stop[:, 2] - start[:, 2])
    cosine = np.cos(start[:, 2])
    sine = np.sin(start[:, 2])
    dx = stop[:, 0] - start[:, 0]
    dy = stop[:, 1] - start[:, 1]
    body_x = cosine * dx + sine * dy
    body_y = -sine * dx + cosine * dy
    return np.stack((body_x, body_y, phi), axis=1)


def invert_relative(relative: np.ndarray) -> np.ndarray:
    """Inverse of an exponentiated relative pose ``(x, y, φ)``."""
    relative = np.asarray(relative, dtype=float)
    cosine = np.cos(relative[:, 2])
    sine = np.sin(relative[:, 2])
    x = -cosine * relative[:, 0] - sine * relative[:, 1]
    y = sine * relative[:, 0] - cosine * relative[:, 1]
    return np.stack((x, y, wrap_angle(-relative[:, 2])), axis=1)


def compose_relative(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compose two exponentiated relative poses, ``first ∘ second``."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    cosine = np.cos(first[:, 2])
    sine = np.sin(first[:, 2])
    x = first[:, 0] + cosine * second[:, 0] - sine * second[:, 1]
    y = first[:, 1] + sine * second[:, 0] + cosine * second[:, 1]
    phi = wrap_angle(first[:, 2] + second[:, 2])
    return np.stack((x, y, phi), axis=1)


def wheel_poses(increments: np.ndarray) -> np.ndarray:
    """Unicycle pose ``Z = Exp(v·dt, 0, ω·dt)`` for each wheel step."""
    increments = np.asarray(increments, dtype=float)
    twist = np.stack((increments[:, 0], np.zeros(len(increments)), increments[:, 1]), axis=1)
    return se2_exp(twist)


def residual_tangents(poses: np.ndarray, increments: np.ndarray) -> np.ndarray:
    """``Log(Z^{-1} T_t^{-1} T_{t+1})``. This is the residual on the unicycle step."""
    leftover = compose_relative(invert_relative(wheel_poses(increments)), relative_poses(poses))
    return se2_log(leftover[:, :2], leftover[:, 2])


def compose(pose, relative) -> np.ndarray:
    """Left-invariant step ``T ∘ Exp(ξ)``, with ``relative`` already exponentiated."""
    pose = np.asarray(pose, dtype=float)
    relative = np.asarray(relative, dtype=float)
    cosine = np.cos(pose[2])
    sine = np.sin(pose[2])
    return np.array(
        [
            pose[0] + cosine * relative[0] - sine * relative[1],
            pose[1] + sine * relative[0] + cosine * relative[1],
            float(wrap_angle(pose[2] + relative[2])),
        ],
        dtype=float,
    )
