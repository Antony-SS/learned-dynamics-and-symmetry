"""Open-loop integration of a predicted SE(2) step.

Wheel increments always come from the log. The path starts at the first Vicon pose.
"""

from __future__ import annotations

import numpy as np
import torch

from unicycle.data import Trajectory
from unicycle.geometry import compose, compose_relative, se2_exp, wheel_poses


def _predict(model: torch.nn.Module, features: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    prediction = model(x).detach().cpu().numpy()
    return prediction


def wheel_open_loop(log: Trajectory) -> tuple[np.ndarray, float]:
    """Integrate the wheel twist ``(v·dt, 0, ω·dt)`` with no learned correction."""
    state = np.asarray(log.poses[0], dtype=float).copy()
    path = np.zeros_like(log.poses)
    path[0] = state
    for index, (v_dt, omega_dt) in enumerate(np.asarray(log.increments, dtype=float)):
        state = compose(state, se2_exp(np.array([v_dt, 0.0, omega_dt])))
        path[index + 1] = state
    error = path[:, :2] - log.poses[:, :2]
    rmse = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
    return path, rmse


def _model_features(model: torch.nn.Module, log: Trajectory) -> np.ndarray:
    """Tick increments when the model asks for them, otherwise the wheel step."""
    if getattr(model, "reads_counts", False):
        if log.counts is None:
            raise ValueError(f"{log.name} has no raw encoder increments")
        features = np.asarray(log.counts, dtype=np.float32)
    else:
        features = np.asarray(log.increments, dtype=np.float32)
    if getattr(model, "uses_pose", False):
        poses = np.asarray(log.poses[:-1], dtype=np.float32)
        features = np.concatenate((features, poses), axis=-1)
    return features


def residual_open_loop(model: torch.nn.Module, log: Trajectory) -> tuple[np.ndarray, float]:
    """Integrate ``T ← T ∘ Z ∘ Exp(r)``, where ``r`` is the network residual."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            increments = np.asarray(log.increments, dtype=np.float32)
            correction = se2_exp(_predict(model, _model_features(model, log)))
            relative = compose_relative(wheel_poses(increments), correction)
            path = _integrate_relative(log, relative)
    finally:
        if was_training:
            model.train()
    error = path[:, :2] - log.poses[:, :2]
    rmse = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
    return path, rmse


def open_loop(model: torch.nn.Module, log: Trajectory) -> tuple[np.ndarray, float]:
    """Integrate from the first Vicon pose. Returns the path and xy RMSE in meters."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            path = _integrate(model, log)
    finally:
        if was_training:
            model.train()
    error = path[:, :2] - log.poses[:, :2]
    rmse = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
    return path, rmse


def _integrate_relative(log: Trajectory, relative: np.ndarray) -> np.ndarray:
    path = np.zeros_like(log.poses)
    path[0] = log.poses[0]
    state = path[0].copy()
    for index in range(len(relative)):
        state = compose(state, relative[index])
        path[index + 1] = state
    return path


def _integrate(model: torch.nn.Module, log: Trajectory) -> np.ndarray:
    steps = se2_exp(_predict(model, _model_features(model, log)))
    return _integrate_relative(log, steps)


def _relative_steps(
    model: torch.nn.Module | None,
    log: Trajectory,
    *,
    residual: bool,
) -> np.ndarray:
    """Relative SE(2) poses for the wheel model or one learned model."""
    wheel = wheel_poses(np.asarray(log.increments, dtype=np.float32))
    if model is None:
        return wheel
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            learned = se2_exp(_predict(model, _model_features(model, log)))
    finally:
        if was_training:
            model.train()
    return compose_relative(wheel, learned) if residual else learned


def position_mse(path: np.ndarray, poses: np.ndarray) -> float:
    """Mean squared Euclidean xy error, in square meters."""
    error = np.asarray(path, dtype=float)[:, :2] - np.asarray(poses, dtype=float)[:, :2]
    return float(np.mean(np.sum(error**2, axis=1)))


def _rollout_sample_count(log: Trajectory, horizon_s: float | None) -> int:
    if horizon_s is None:
        return log.n_poses
    if horizon_s <= 0:
        raise ValueError(f"horizon_s must be positive, got {horizon_s}")
    width = int(round(float(horizon_s) / log.dt))
    if width % 2 == 0:
        width += 1
    steps = int(log.increments.shape[0])
    if width < 1 or width > steps:
        raise ValueError(
            f"{log.name} has {steps} steps, fewer than a {horizon_s:g} s window ({width} steps)"
        )
    return (steps - width + 1) * width


def rollout_position_mse(
    model: torch.nn.Module | None,
    log: Trajectory,
    *,
    residual: bool = False,
    horizon_s: float | None = None,
) -> float:
    """Full or fixed-horizon open-loop xy MSE.

    ``horizon_s=None`` starts once at the beginning of the complete log.
    A finite horizon starts at every possible Vicon pose, rolls for the nearest
    odd number of steps to that duration, and averages squared Euclidean xy
    error over every predicted pose in every window. Passing ``model=None``
    evaluates the wheel unicycle.
    """
    relative = _relative_steps(model, log, residual=residual)
    if horizon_s is None:
        path = _integrate_relative(log, relative)
        return position_mse(path, log.poses)
    width = int(round(float(horizon_s) / log.dt))
    if width % 2 == 0:
        width += 1
    _rollout_sample_count(log, horizon_s)
    total = 0.0
    count = 0
    for start in range(len(relative) - width + 1):
        state = np.asarray(log.poses[start], dtype=float)
        for offset, step in enumerate(relative[start : start + width], start=1):
            state = compose(state, step)
            error = state[:2] - log.poses[start + offset, :2]
            total += float(error @ error)
            count += 1
    return total / count


def pooled_rollout_position_rmse(
    model: torch.nn.Module | None,
    logs: list[Trajectory],
    *,
    residual: bool = False,
    horizon_s: float | None = None,
) -> float:
    """Sample-weighted position RMSE over multiple full or fixed-horizon rollouts."""
    total = 0.0
    count = 0
    for log in logs:
        samples = _rollout_sample_count(log, horizon_s)
        mse = rollout_position_mse(model, log, residual=residual, horizon_s=horizon_s)
        total += mse * samples
        count += samples
    if count == 0:
        raise ValueError("need at least one rollout sample")
    return float(np.sqrt(total / count))


def held_out_heading_rmse(model: torch.nn.Module, logs: list[Trajectory], *, residual: bool = False) -> float:
    """Length-weighted RMSE of wrapped heading, in radians, across logs."""
    rollout = residual_open_loop if residual else open_loop
    total = 0.0
    weight = 0
    for log in logs:
        path, _rmse = rollout(model, log)
        error = (path[:, 2] - log.poses[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
        total += float(np.sum(error**2))
        weight += int(error.shape[0])
    return float(np.sqrt(total / weight))


def held_out_rmse(model: torch.nn.Module, logs: list[Trajectory], *, residual: bool = False) -> float:
    """Length-weighted xy RMSE across held-out logs."""
    rollout = residual_open_loop if residual else open_loop
    total = 0.0
    weight = 0.0
    for log in logs:
        _path, rmse = rollout(model, log)
        total += (rmse**2) * log.n_poses
        weight += log.n_poses
    return float(np.sqrt(total / weight))
