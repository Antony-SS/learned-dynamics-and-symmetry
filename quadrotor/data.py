"""NanoBench flight loading.

CSVs are fetched into ``data/nanobench`` and are not part of this repository.
Euler angles in the files are radians (they match the quaternion), despite the
upstream README listing degrees. Accelerometer readings are converted from g
to m/s². Motor commands are scaled from PWM counts into ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from quadrotor.geometry import quat_to_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data" / "nanobench" / "datasets" / "dataset"
GRAVITY = 9.80665
PWM_FULL_SCALE = 65535.0

# NanoBench Task 1. Category A is the multisine excitation. Category B is the
# held-out tracking set in Table III. The last fifth of each multisine flight
# is validation, and no window is allowed to cross that cut.
MULTISINE_PREFIX = "A1b_multisine_sysid"
TRAIN_FRACTION = 0.8

_COLUMNS = (
    "t",
    "px",
    "py",
    "pz",
    "qx",
    "qy",
    "qz",
    "qw",
    "vx",
    "vy",
    "vz",
    "wx_vicon",
    "wy_vicon",
    "wz_vicon",
    "imu_acc_x",
    "imu_acc_y",
    "imu_acc_z",
    "imu_gyro_x",
    "imu_gyro_y",
    "imu_gyro_z",
    "motor_motor_m1",
    "motor_motor_m2",
    "motor_motor_m3",
    "motor_motor_m4",
    "att_stateEstimate_qx",
    "att_stateEstimate_qy",
    "att_stateEstimate_qz",
    "att_stateEstimate_qw",
    "pwr_pm_vbat",
)


@dataclass
class Flight:
    name: str
    time: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    velocity_world: np.ndarray
    angular_velocity_world: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    pwm: np.ndarray
    voltage: np.ndarray
    ekf_rotation: np.ndarray
    path: str

    @property
    def n_poses(self) -> int:
        return int(self.position.shape[0])


def load_flight(path: str | Path) -> Flight:
    path = Path(path)
    frame = pd.read_csv(path)
    missing = [column for column in _COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    values = frame.loc[:, list(_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} has non-finite samples")
    if len(values) < 2:
        raise ValueError(f"{path} has {len(values)} samples")
    rotation = quat_to_matrix(values[:, 4:8])
    # Columns are t, px py pz, qx qy qz qw, then 12 signals, then four motors.
    ekf_quat = values[:, 24:28]
    return Flight(
        name=path.stem,
        time=values[:, 0],
        position=values[:, 1:4],
        rotation=rotation,
        velocity_world=values[:, 8:11],
        angular_velocity_world=values[:, 11:14],
        accel=values[:, 14:17] * GRAVITY,
        gyro=values[:, 17:20],
        pwm=values[:, 20:24] / PWM_FULL_SCALE,
        voltage=values[:, 28],
        ekf_rotation=quat_to_matrix(ekf_quat),
        path=str(path),
    )


def dataset_dir(root: str | Path | None = None) -> Path:
    root = DEFAULT_DATASET if root is None else Path(root)
    if not root.is_dir() or not any(root.glob("*.csv")):
        raise FileNotFoundError(
            f"no NanoBench CSVs under {root}. Run `python -m quadrotor.fetch_nanobench` from the repo root."
        )
    return root


def _slice(flight: Flight, start: int, stop: int, suffix: str) -> Flight:
    """Poses ``[start, stop)``. The step across ``start`` is in neither piece."""
    if stop - start < 2:
        raise ValueError(f"{flight.name} slice [{start}:{stop}] is shorter than two poses")
    return Flight(
        name=f"{flight.name} {suffix}",
        time=flight.time[start:stop],
        position=flight.position[start:stop],
        rotation=flight.rotation[start:stop],
        velocity_world=flight.velocity_world[start:stop],
        angular_velocity_world=flight.angular_velocity_world[start:stop],
        gyro=flight.gyro[start:stop],
        accel=flight.accel[start:stop],
        pwm=flight.pwm[start:stop],
        voltage=flight.voltage[start:stop],
        ekf_rotation=flight.ekf_rotation[start:stop],
        path=flight.path,
    )


def load_split(
    root: str | Path | None = None,
) -> tuple[list[Flight], list[Flight], list[Flight]]:
    """NanoBench Task 1 split: multisine 80/20, then every Category B flight.

    The first 80% of each multisine recording is training and the last 20% is
    validation. The two pieces do not share a step, so a window built inside
    one piece cannot cross the cut. Category B tracking flights are the test
    set and are not used for training. Category C battery-drain hovers are left
    out.
    """
    root = dataset_dir(root)
    files = sorted(root.glob("*.csv"))
    multisine = [path for path in files if path.stem.startswith(MULTISINE_PREFIX)]
    test_paths = [path for path in files if path.stem.startswith("B")]
    if not multisine:
        raise FileNotFoundError(f"no {MULTISINE_PREFIX} flights under {root}")
    if not test_paths:
        raise FileNotFoundError(f"no Category B flights under {root}")
    train: list[Flight] = []
    validation: list[Flight] = []
    for path in multisine:
        flight = load_flight(path)
        cut = int(flight.n_poses * TRAIN_FRACTION)
        train.append(_slice(flight, 0, cut, "train"))
        validation.append(_slice(flight, cut, flight.n_poses, "val"))
    test = [load_flight(path) for path in test_paths]
    return train, validation, test


def load_data_rich_split(
    root: str | Path | None = None,
    *,
    seed: int = 0,
) -> tuple[list[Flight], list[Flight], list[Flight]]:
    """Category A plus a flight-level, trajectory-stratified Category B split.

    Every available multisine flight is training data. Category B is grouped by
    trajectory family (circle, figure-eight, trefoil, and so on), shuffled with
    a fixed seed, then split approximately 70/15/15. A complete recording
    belongs to exactly one split, preventing overlapping-window leakage.
    """
    root = dataset_dir(root)
    files = sorted(root.glob("*.csv"))
    a_paths = [path for path in files if path.stem.startswith(MULTISINE_PREFIX)]
    b_paths = [path for path in files if path.stem.startswith("B")]
    if not a_paths or not b_paths:
        raise FileNotFoundError("the data-rich split requires Categories A and B")

    groups: dict[str, list[Path]] = {}
    for path in b_paths:
        match = re.match(r"(B\d+)_", path.stem)
        if match is None:
            raise ValueError(f"cannot identify trajectory family for {path.name}")
        groups.setdefault(match.group(1), []).append(path)

    rng = np.random.default_rng(seed)
    train_paths = list(a_paths)
    validation_paths: list[Path] = []
    test_paths: list[Path] = []
    for family in sorted(groups):
        paths = sorted(groups[family])
        paths = [paths[index] for index in rng.permutation(len(paths))]
        if len(paths) == 1:
            test_paths.extend(paths)
            continue
        n_test = max(1, int(round(0.15 * len(paths))))
        n_validation = max(1, int(round(0.15 * len(paths)))) if len(paths) >= 3 else 0
        n_train = len(paths) - n_validation - n_test
        train_paths.extend(paths[:n_train])
        validation_paths.extend(paths[n_train : n_train + n_validation])
        test_paths.extend(paths[n_train + n_validation :])

    return (
        [load_flight(path) for path in train_paths],
        [load_flight(path) for path in validation_paths],
        [load_flight(path) for path in test_paths],
    )
