"""Robotarium trajectory loading.

Logs live in ``robotarium-capstone-data`` next to this repo. Only poses and
wheel encoders are read. Compass, IMU, and factor-graph fields are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Wheel geometry used by the robotarium encoder-to-unicycle map.
WHEEL_RADIUS = 0.016
BASE_LENGTH = 0.105
ENCODER_COUNTS_PER_REVOLUTION = 28.0
MOTOR_GEAR_RATIO = 100.37

# Actuator caps from ARobotarium. The angular cap is the yaw rate at the
# linear cap with the wheels opposed: 2 * v_max / robot_diameter.
TIME_STEP = 0.033
MAX_LINEAR_VELOCITY = 0.2
ROBOT_DIAMETER = 0.11
MAX_ANGULAR_VELOCITY = 2.0 * MAX_LINEAR_VELOCITY / ROBOT_DIAMETER
# Encoder stutter left out of the loss and the diagnostic plots. One half is a
# window past the actuator cap. The other half reads about zero while the
# command was moving. A real stop has both near zero and stays in.
ENCODER_IDLE_SPEED = 0.02
COMMAND_MOVING_SPEED = 0.05
# Encoder counts stutter by a step or two. Sum this many raw steps, starting
# at the first pose, and learn the Vicon motion over that same window.
WINDOW = 3

# Logs live next to this repo, not inside it: capstone/robotarium-capstone-data.
# ``standard_collections`` is the official robot 11 set. ``asymmetry_introduced``
# is a different experiment and is not part of this split.
DEFAULT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "robotarium-capstone-data"
    / "single_robots"
    / "data_robot_11"
    / "standard_collections"
)
# Last quarter of each training collection is the validation set used to pick
# checkpoints. These three collections are held out whole and never enter training.
HELD_OUT_FRACTION = 0.25
TRAIN_LOGS = (
    "dynamic_slide",
    "lawnmower_diagonal",
    "lawnmower_horizontal",
    "lawnmower_horizontal_reverse",
    "lawnmower_vertical",
    "lawnmower_vertical_reverse",
    "spiral",
    "spiral_reverse",
)
TEST_LOGS = (
    "figure8",
    "lawnmower_diagonal_reverse",
    "random_trajectory",
)


@dataclass
class Trajectory:
    name: str
    poses: np.ndarray
    increments: np.ndarray
    commands: np.ndarray
    dt: float
    path: str
    # Raw left/right encoder tick deltas. ``increments`` is the unicycle twist
    # derived from these. Learned models read ``counts``.
    counts: np.ndarray | None = None

    @property
    def n_poses(self) -> int:
        return int(self.poses.shape[0])


def kept_steps(log: Trajectory) -> np.ndarray:
    """True for wheel steps that stay in the loss and the diagnostic plots."""
    speed = np.asarray(log.increments, dtype=float) / log.dt
    command = np.asarray(log.commands, dtype=float)
    over_speed = np.abs(speed[:, 0]) > MAX_LINEAR_VELOCITY
    over_yaw = np.abs(speed[:, 1]) > MAX_ANGULAR_VELOCITY
    dropped = (np.abs(speed[:, 0]) < ENCODER_IDLE_SPEED) & (np.abs(command[:, 0]) > COMMAND_MOVING_SPEED)
    return ~(over_speed | over_yaw | dropped)


def encoder_delta(count_prev, count_next) -> np.ndarray:
    """Raw left/right tick increments. Shape matches ``count_prev``."""
    return np.asarray(count_next, dtype=float) - np.asarray(count_prev, dtype=float)


METERS_PER_TICK = (2.0 * np.pi * WHEEL_RADIUS) / (
    ENCODER_COUNTS_PER_REVOLUTION * MOTOR_GEAR_RATIO
)
_WHEEL_SPEED_CAP = MAX_LINEAR_VELOCITY + MAX_ANGULAR_VELOCITY * BASE_LENGTH / 2.0
COUNT_LIMIT = _WHEEL_SPEED_CAP * TIME_STEP * WINDOW / METERS_PER_TICK


def body_increment(count_prev, count_next) -> np.ndarray:
    """Body step ``(v·dt, ω·dt)`` from consecutive left/right encoder counts."""
    delta = encoder_delta(count_prev, count_next)
    arc = delta * METERS_PER_TICK
    left = arc[..., 0]
    right = arc[..., 1]
    v_dt = 0.5 * (left + right)
    omega_dt = (right - left) / BASE_LENGTH
    return np.stack((v_dt, omega_dt), axis=-1)


def load_trajectory(path: str | Path) -> Trajectory:
    path = Path(path)
    log = np.load(path)
    for key in ("poses", "encoders", "commanded_velocities", "time_step"):
        if key not in log.files:
            raise KeyError(f"{path} is missing '{key}'")
    poses = np.asarray(log["poses"], dtype=float)
    encoders = np.asarray(log["encoders"], dtype=float)
    commands = np.asarray(log["commanded_velocities"], dtype=float)
    if poses.shape[0] != 3:
        raise ValueError(f"poses must have shape (3, N), got {poses.shape}")
    if encoders.shape != (2, poses.shape[1]):
        raise ValueError(f"encoders must have shape (2, N), got {encoders.shape}")
    if commands.shape != (2, poses.shape[1]):
        raise ValueError(f"commanded_velocities must have shape (2, N), got {commands.shape}")
    gt = poses.T
    if len(gt) < 2:
        raise ValueError(f"{path} has {len(gt)} poses")
    counts = encoder_delta(encoders.T[:-1], encoders.T[1:])
    increments = body_increment(encoders.T[:-1], encoders.T[1:])
    return Trajectory(
        name=path.parent.name,
        poses=gt,
        increments=np.asarray(increments, dtype=float),
        commands=commands.T[:-1],
        dt=float(log["time_step"]),
        path=str(path),
        counts=np.asarray(counts, dtype=float),
    )


def coarsen(log: Trajectory, window: int = WINDOW) -> Trajectory:
    """Sum ``window`` encoder steps and keep every ``window``-th pose.

    Windows start at the first pose. A trailing partial window is dropped.
    The new wheel step is the sum of ``(v·dt, ω·dt)`` over the window, and the
    kept poses make the label ``Log(T_t^{-1} T_{t+window})``.
    """
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    steps = int(log.increments.shape[0])
    kept = steps // window
    if kept < 1:
        raise ValueError(f"{log.name} has {steps} steps, fewer than a window of {window}")
    block = log.increments[: kept * window].reshape(kept, window, -1).sum(axis=1)
    command = log.commands[: kept * window].reshape(kept, window, -1).mean(axis=1)
    counts = None
    if log.counts is not None:
        counts = log.counts[: kept * window].reshape(kept, window, -1).sum(axis=1)
    return Trajectory(
        name=log.name,
        poses=log.poses[: kept * window + 1 : window],
        increments=block,
        commands=command,
        dt=log.dt * window,
        path=log.path,
        counts=counts,
    )


def _segment(log: Trajectory, start: int, stop: int, label: str) -> Trajectory:
    """Poses ``[start, stop)`` and the wheel steps that connect them."""
    poses = log.poses[start:stop]
    increments = log.increments[start : stop - 1]
    commands = log.commands[start : stop - 1]
    counts = None if log.counts is None else log.counts[start : stop - 1]
    if len(poses) < 2 or len(increments) != len(poses) - 1 or len(commands) != len(increments):
        raise ValueError(f"bad segment {log.name} [{start}:{stop}] from {log.n_poses} poses")
    return Trajectory(
        name=f"{log.name} ({label})",
        poses=poses,
        increments=increments,
        commands=commands,
        dt=log.dt,
        path=log.path,
        counts=counts,
    )


def _held_out_cut(steps: int) -> int:
    """First validation step. The last quarter of ``steps`` starts here."""
    held_steps = int(round(steps * HELD_OUT_FRACTION))
    held_steps = min(max(held_steps, 1), steps - 1)
    return steps - held_steps


def _load_named(root: str | Path) -> dict[str, Trajectory]:
    """Coarsened official collections, keyed by folder name."""
    root = Path(root)
    paths = sorted(root.glob("*/*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"no Robotarium logs under {root}. Expected the existing checkout at {DEFAULT_ROOT}."
        )
    found: dict[str, Trajectory] = {}
    for path in paths:
        log = coarsen(load_trajectory(path))
        if log.name in found:
            raise ValueError(f"two logs named {log.name} under {root}")
        found[log.name] = log
    expected = set(TRAIN_LOGS) | set(TEST_LOGS)
    missing = expected - found.keys()
    extra = found.keys() - expected
    if missing or extra:
        raise FileNotFoundError(
            f"collections under {root} do not match the official split. "
            f"missing {sorted(missing) or 'none'}, extra {sorted(extra) or 'none'}"
        )
    return found


def _train_val(log: Trajectory) -> tuple[Trajectory, Trajectory]:
    """First 75% trains. Last 25% is the validation segment."""
    steps = int(log.increments.shape[0])
    cut = _held_out_cut(steps)
    return (
        _segment(log, 0, cut + 1, "first 75%"),
        _segment(log, cut, log.n_poses, "last 25%"),
    )


def load_efficiency_splits(
    train_fractions: tuple[float, ...] | list[float] = (0.75, 0.55, 0.35),
    root: str | Path = DEFAULT_ROOT,
) -> tuple[list[Trajectory], list[tuple[float, list[Trajectory]]], list[Trajectory]]:
    """Same last-quarter validation. Each fraction is a prefix of every training log.

    ``0.75`` is the official training split. A smaller fraction ends earlier in
    every collection, so the maneuver mix stays the same and only the length
    changes. Steps between that prefix and the last quarter are unused, so
    checkpoint selection does not change. A prefix cannot pass the 75% cut.
    The returned ``full_train`` is the original 75% split, used to scale
    epochs by the training steps that were cut. The three test collections
    are not included.
    """
    if not train_fractions:
        raise ValueError("need at least one training prefix")
    limit = 1.0 - HELD_OUT_FRACTION
    fractions = [float(fraction) for fraction in train_fractions]
    for fraction in fractions:
        if not 0.0 < fraction <= limit + 1e-9:
            raise ValueError(f"training prefix must be in (0, {limit}], got {fraction}")
    found = _load_named(root)
    logs = [found[name] for name in TRAIN_LOGS]
    held_out = []
    all_train = []
    for log in logs:
        train_log, val_log = _train_val(log)
        all_train.append(train_log)
        held_out.append(val_log)
    splits: list[tuple[float, list[Trajectory]]] = []
    for fraction in fractions:
        prefixes = []
        for log in logs:
            steps = int(log.increments.shape[0])
            cut = _held_out_cut(steps)
            if abs(fraction - limit) <= 1e-9:
                stop = cut
            else:
                stop = min(max(int(round(steps * fraction)), 1), cut)
            prefixes.append(_segment(log, 0, stop + 1, f"first {100 * fraction:.0f}%"))
        splits.append((fraction, prefixes))
    return held_out, splits, all_train


def load_split(
    root: str | Path = DEFAULT_ROOT,
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory]]:
    """Training prefix, validation quarter, and the three full test collections.

    Each log is coarsened into windows of ``WINDOW`` raw steps before any cut,
    so a window does not cross the training/validation boundary. The test
    collections are figure 8, diagonal lawnmower reverse, and random trajectory.
    They are returned whole. Checkpoints are chosen on the validation quarter
    of the other eight collections.
    """
    found = _load_named(root)
    train: list[Trajectory] = []
    validation: list[Trajectory] = []
    for name in TRAIN_LOGS:
        train_log, val_log = _train_val(found[name])
        train.append(train_log)
        validation.append(val_log)
    test = [found[name] for name in TEST_LOGS]
    return train, validation, test
