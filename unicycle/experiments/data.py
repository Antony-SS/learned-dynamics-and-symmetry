"""Robotarium collections. Poses and commanded (v, ω) only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from unicycle.geometry import wrap_angle

TIME_STEP = 0.033
TRIM_S = 0.5
MAX_LINEAR_VELOCITY = 0.2
HELD_OUT_FRACTION = 0.25

DEFAULT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "robotarium-capstone-data"
    / "single_robots"
    / "data_robot_11"
    / "standard_collections"
)

TRAIN_LOGS = (
    "arc_straight",
    "figure8_vertical",
    "lawnmower_horizontal",
    "lawnmower_vertical",
    "random_trajectory",
    "spiral",
    "spiral_cw",
)
TEST_LOGS = (
    "figure8",
    "lawnmower_diagonal",
    "slalom",
)


def body_rates(poses: np.ndarray, dt: float) -> np.ndarray:
    start, stop = poses[:-1], poses[1:]
    dtheta = wrap_angle(stop[:, 2] - start[:, 2])
    cosine = np.cos(start[:, 2])
    sine = np.sin(start[:, 2])
    dx = stop[:, 0] - start[:, 0]
    dy = stop[:, 1] - start[:, 1]
    v = (cosine * dx + sine * dy) / dt
    omega = dtheta / dt
    return np.stack((v, omega), axis=-1)


@dataclass
class Trajectory:
    name: str
    poses: np.ndarray
    rates: np.ndarray
    commands: np.ndarray
    dt: float
    path: str

    @property
    def n_poses(self) -> int:
        return int(self.poses.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.commands.shape[0])


def load_trajectory(path: str | Path) -> Trajectory:
    path = Path(path)
    log = np.load(path)
    for key in ("poses", "commanded_velocities", "time_step"):
        if key not in log.files:
            raise KeyError(f"{path} is missing '{key}'")
    poses = np.asarray(log["poses"], dtype=float)
    commands = np.asarray(log["commanded_velocities"], dtype=float)
    if poses.shape[0] != 3:
        raise ValueError(f"poses must have shape (3, N), got {poses.shape}")
    if commands.shape != (2, poses.shape[1]):
        raise ValueError(f"commanded_velocities must have shape (2, N), got {commands.shape}")
    gt = poses.T
    dt = float(log["time_step"])
    if len(gt) < 2:
        raise ValueError(f"{path} has {len(gt)} poses")
    commands = commands.T[:-1]
    skip = int(round(TRIM_S / dt)) if TRIM_S > 0 else 0
    if skip:
        if len(gt) - skip < 2:
            raise ValueError(f"{path} has {len(gt)} poses, fewer than a {TRIM_S:g} s trim")
        gt = gt[skip:]
        commands = commands[skip:]
    return Trajectory(
        name=path.parent.name,
        poses=gt,
        rates=body_rates(gt, dt),
        commands=np.asarray(commands, dtype=float),
        dt=dt,
        path=str(path),
    )


def mirror_log(log: Trajectory) -> Trajectory:
    """Left–right copy: y and θ flip, ω and ω_cmd flip."""
    poses = np.asarray(log.poses, dtype=float).copy()
    poses[:, 1] *= -1.0
    poses[:, 2] = wrap_angle(-poses[:, 2])
    rates = np.asarray(log.rates, dtype=float).copy()
    rates[:, 1] *= -1.0
    commands = np.asarray(log.commands, dtype=float).copy()
    commands[:, 1] *= -1.0
    return Trajectory(
        name=f"{log.name} (mirror)",
        poses=poses,
        rates=rates,
        commands=commands,
        dt=log.dt,
        path=log.path,
    )


def _segment(log: Trajectory, start: int, stop: int, label: str) -> Trajectory:
    poses = log.poses[start:stop]
    return Trajectory(
        name=f"{log.name} ({label})",
        poses=poses,
        rates=log.rates[start : stop - 1],
        commands=log.commands[start : stop - 1],
        dt=log.dt,
        path=log.path,
    )


def _held_out_cut(steps: int) -> int:
    held = min(max(int(round(steps * HELD_OUT_FRACTION)), 1), steps - 1)
    return steps - held


def load_named(root: str | Path = DEFAULT_ROOT) -> dict[str, Trajectory]:
    root = Path(root)
    paths = sorted(root.glob("*/*.npz"))
    if not paths:
        raise FileNotFoundError(f"no Robotarium logs under {root}")
    found: dict[str, Trajectory] = {}
    for path in paths:
        log = load_trajectory(path)
        found[log.name] = log
    expected = set(TRAIN_LOGS) | set(TEST_LOGS)
    missing = expected - found.keys()
    if missing:
        raise FileNotFoundError(f"missing collections under {root}: {sorted(missing)}")
    return {name: found[name] for name in expected}


def load_split(
    root: str | Path = DEFAULT_ROOT,
    *,
    augment_train: bool = False,
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory]]:
    """First 75% of each train collection, last 25% val, three full test logs.

    Checkpoints are chosen on the validation quarter (5 s xyyaw). Test is
    figure 8, diagonal lawnmower, and slalom. ``augment_train`` appends the
    left–right mirror of every training prefix.
    """
    found = load_named(root)
    train: list[Trajectory] = []
    validation: list[Trajectory] = []
    for name in TRAIN_LOGS:
        steps = found[name].n_steps
        cut = _held_out_cut(steps)
        train.append(_segment(found[name], 0, cut + 1, "first 75%"))
        validation.append(_segment(found[name], cut, found[name].n_poses, "last 25%"))
    if augment_train:
        train = train + [mirror_log(log) for log in train]
    test = [found[name] for name in TEST_LOGS]
    return train, validation, test


def turn_counts(logs: list[Trajectory], eps: float = 0.05) -> dict[str, int]:
    omega = np.concatenate([log.commands[:, 1] for log in logs])
    return {
        "left": int(np.sum(omega > eps)),
        "right": int(np.sum(omega < -eps)),
        "straight": int(np.sum(np.abs(omega) <= eps)),
        "steps": int(omega.size),
    }


def commanded_vw(logs: list[Trajectory]) -> tuple[np.ndarray, np.ndarray]:
    v = np.concatenate([log.commands[:, 0] for log in logs])
    w = np.concatenate([log.commands[:, 1] for log in logs])
    return v, w
