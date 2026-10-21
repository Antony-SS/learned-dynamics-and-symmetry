"""Open-loop integration of a predicted SE(2) step.

Wheel increments always come from the log. The path starts at the first Vicon pose.
"""

from __future__ import annotations

import numpy as np
import torch

from unicycle.data import Trajectory
from unicycle.geometry import body_steps, compose, compose_relative, se2_exp, wheel_poses
from unicycle.models import history_slice, rollout_command_velocity


def _module_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return next(model.buffers()).device


def _predict(model: torch.nn.Module, features: np.ndarray) -> np.ndarray:
    device = _module_device(model)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    prediction = model(x).detach().cpu().numpy()
    return prediction


def _seed_velocity(model: torch.nn.Module, log: Trajectory, start: int) -> np.ndarray:
    dt = float(log.dt)
    encoder = np.asarray(log.increments, dtype=np.float32) / dt
    vicon = body_steps(np.asarray(log.poses, dtype=float)) / dt
    if getattr(model, "has_sideslip", False):
        measured = np.stack((encoder[:, 0], vicon[:, 1], encoder[:, 1]), axis=1).astype(np.float32)
    elif getattr(model, "seeds_vicon", False):
        measured = np.stack((vicon[:, 0], vicon[:, 2]), axis=1).astype(np.float32)
    else:
        measured = encoder
    if start > 0:
        return measured[start - 1]
    return measured[start]


def _command_lag_relative(model: torch.nn.Module, log: Trajectory, start: int, steps: int) -> np.ndarray:
    """Relative SE(2) steps from a re-seeded velocity rollout."""
    device = _module_device(model)
    dt = float(log.dt)
    commands = np.asarray(log.commands[start : start + steps], dtype=np.float32)
    nu0 = torch.as_tensor(_seed_velocity(model, log, start), device=device).unsqueeze(0)
    command = torch.as_tensor(commands, device=device).unsqueeze(0)
    hidden = None
    if getattr(model, "uses_hidden", False):
        encoder = np.asarray(log.increments, dtype=np.float32) / dt
        nu_h, u_h = history_slice(
            encoder,
            np.asarray(log.commands, dtype=np.float32),
            start,
            int(model.history_steps(dt)),
        )
        hidden = model.warm_hidden(
            torch.as_tensor(nu_h, device=device).unsqueeze(0),
            torch.as_tensor(u_h, device=device).unsqueeze(0),
        )
    poses = None
    if getattr(model, "uses_pose", False):
        poses = torch.as_tensor(
            np.asarray(log.poses[start : start + steps], dtype=np.float32),
            device=device,
        ).unsqueeze(0)
    velocity = rollout_command_velocity(
        model, nu0, command, dt, hidden=hidden, poses=poses
    ).squeeze(0).detach().cpu().numpy()
    if velocity.shape[-1] == 3:
        twist = np.stack((velocity[:, 0] * dt, velocity[:, 1] * dt, velocity[:, 2] * dt), axis=1)
    else:
        twist = np.stack((velocity[:, 0] * dt, np.zeros(steps), velocity[:, 1] * dt), axis=1)
    return se2_exp(twist)


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


def command_open_loop(log: Trajectory) -> tuple[np.ndarray, float]:
    """Integrate the commanded twist ``(v·dt, 0, ω·dt)`` with no learned correction."""
    commands = np.asarray(log.commands, dtype=float)
    increments = np.stack((commands[:, 0] * log.dt, commands[:, 1] * log.dt), axis=1)
    state = np.asarray(log.poses[0], dtype=float).copy()
    path = np.zeros_like(log.poses)
    path[0] = state
    for index, (v_dt, omega_dt) in enumerate(increments):
        state = compose(state, se2_exp(np.array([v_dt, 0.0, omega_dt])))
        path[index + 1] = state
    error = path[:, :2] - log.poses[:, :2]
    rmse = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
    return path, rmse


def _model_features(model: torch.nn.Module, log: Trajectory) -> np.ndarray:
    """Commanded (v, ω), tick increments, or the wheel step."""
    if getattr(model, "reads_command", False):
        features = np.asarray(log.commands, dtype=np.float32)
    elif getattr(model, "reads_counts", False):
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
    if getattr(model, "uses_velocity", False):
        steps = _command_lag_relative(model, log, 0, int(log.increments.shape[0]))
        return _integrate_relative(log, steps)
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
            if getattr(model, "uses_velocity", False):
                learned = _command_lag_relative(model, log, 0, int(log.increments.shape[0]))
            else:
                learned = se2_exp(_predict(model, _model_features(model, log)))
    finally:
        if was_training:
            model.train()
    return compose_relative(wheel, learned) if residual else learned


def position_mse(path: np.ndarray, poses: np.ndarray) -> float:
    """Mean squared Euclidean xy error, in square meters."""
    error = np.asarray(path, dtype=float)[:, :2] - np.asarray(poses, dtype=float)[:, :2]
    return float(np.mean(np.sum(error**2, axis=1)))


def _window_width(log: Trajectory, horizon_s: float) -> int:
    """Odd step count nearest ``horizon_s``, or the full log if that is shorter."""
    if horizon_s <= 0:
        raise ValueError(f"horizon_s must be positive, got {horizon_s}")
    steps = int(log.increments.shape[0])
    if steps < 1:
        raise ValueError(f"{log.name} has no steps")
    width = int(round(float(horizon_s) / float(log.dt)))
    if width % 2 == 0:
        width += 1
    if width > steps:
        return steps
    if width < 1:
        raise ValueError(f"{log.name} has {steps} steps, shorter than a {horizon_s:g} s window")
    return width


def _rollout_sample_count(log: Trajectory, horizon_s: float | None) -> int:
    if horizon_s is None:
        return log.n_poses
    width = _window_width(log, horizon_s)
    steps = int(log.increments.shape[0])
    return (steps - width + 1) * width


def rollout_position_mse(
    model: torch.nn.Module | None,
    log: Trajectory,
    *,
    residual: bool = False,
    horizon_s: float | None = None,
    start_stride: int = 1,
) -> float:
    """Full or fixed-horizon open-loop xy MSE.

    ``horizon_s=None`` starts once at the beginning of the complete log.
    A finite horizon starts at every possible Vicon pose, rolls for the nearest
    odd number of steps to that duration, and averages squared Euclidean xy
    error over every predicted pose in every window. Passing ``model=None``
    evaluates the wheel unicycle. A log shorter than the horizon uses the
    full recording. ``start_stride`` skips starts when scoring many windows.
    """
    if start_stride < 1:
        raise ValueError(f"start_stride must be positive, got {start_stride}")
    if model is not None and getattr(model, "uses_velocity", False) and horizon_s is not None:
        width = _window_width(log, horizon_s)
        was_training = model.training
        model.eval()
        total = 0.0
        count = 0
        try:
            with torch.no_grad():
                last = int(log.increments.shape[0]) - width + 1
                for start in range(0, last, start_stride):
                    relative = _command_lag_relative(model, log, start, width)
                    state = np.asarray(log.poses[start], dtype=float)
                    for offset, step in enumerate(relative, start=1):
                        state = compose(state, step)
                        error = state[:2] - log.poses[start + offset, :2]
                        total += float(error @ error)
                        count += 1
        finally:
            if was_training:
                model.train()
        if count == 0:
            raise ValueError(f"{log.name} has no {horizon_s:g} s rollout windows")
        return total / count
    relative = _relative_steps(model, log, residual=residual)
    if horizon_s is None:
        path = _integrate_relative(log, relative)
        return position_mse(path, log.poses)
    width = _window_width(log, horizon_s)
    total = 0.0
    count = 0
    last = len(relative) - width + 1
    for start in range(0, last, start_stride):
        state = np.asarray(log.poses[start], dtype=float)
        for offset, step in enumerate(relative[start : start + width], start=1):
            state = compose(state, step)
            error = state[:2] - log.poses[start + offset, :2]
            total += float(error @ error)
            count += 1
    if count == 0:
        raise ValueError(f"{log.name} has no {horizon_s:g} s rollout windows")
    return total / count


def pooled_rollout_position_mse(
    model: torch.nn.Module | None,
    logs: list[Trajectory],
    *,
    residual: bool = False,
    horizon_s: float | None = None,
    start_stride: int = 1,
) -> float:
    """Sample-weighted position MSE over multiple full or fixed-horizon rollouts."""
    total = 0.0
    count = 0
    for log in logs:
        mse = rollout_position_mse(
            model, log, residual=residual, horizon_s=horizon_s, start_stride=start_stride
        )
        if horizon_s is None:
            samples = log.n_poses
        else:
            width = _window_width(log, horizon_s)
            starts = range(0, int(log.increments.shape[0]) - width + 1, start_stride)
            samples = len(list(starts)) * width
        total += mse * samples
        count += samples
    if count == 0:
        raise ValueError("need at least one rollout sample")
    return float(total / count)


def pooled_rollout_position_rmse(
    model: torch.nn.Module | None,
    logs: list[Trajectory],
    *,
    residual: bool = False,
    horizon_s: float | None = None,
    start_stride: int = 1,
) -> float:
    """Sample-weighted position RMSE over multiple full or fixed-horizon rollouts."""
    return float(
        np.sqrt(
            pooled_rollout_position_mse(
                model,
                logs,
                residual=residual,
                horizon_s=horizon_s,
                start_stride=start_stride,
            )
        )
    )


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


def held_out_mse(model: torch.nn.Module, logs: list[Trajectory], *, residual: bool = False) -> float:
    """Length-weighted xy MSE across whole-log open-loop rollouts."""
    rollout = residual_open_loop if residual else open_loop
    total = 0.0
    weight = 0.0
    for log in logs:
        _path, rmse = rollout(model, log)
        total += (rmse**2) * log.n_poses
        weight += log.n_poses
    if weight == 0:
        raise ValueError("need at least one pose")
    return float(total / weight)


def held_out_rmse(model: torch.nn.Module, logs: list[Trajectory], *, residual: bool = False) -> float:
    """Length-weighted xy RMSE across held-out logs."""
    return float(np.sqrt(held_out_mse(model, logs, residual=residual)))
