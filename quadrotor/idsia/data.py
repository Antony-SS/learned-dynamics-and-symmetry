"""Official IDSIA nano-drone benchmark split and CSV conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quadrotor.canonical import PreparedFlight
from quadrotor.geometry import quat_to_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(__file__).resolve().parent / "nanodrone-sysid-benchmark"
TRAIN_TRAJECTORIES = ("square", "random", "chirp")

_COLUMNS = (
    "t",
    "x", "y", "z",
    "qx", "qy", "qz", "qw",
    "vx", "vy", "vz",
    "wx", "wy", "wz",
    "m1_rads", "m2_rads", "m3_rads", "m4_rads",
)


def _load(path: Path) -> PreparedFlight:
    frame = pd.read_csv(path)
    missing = [column for column in _COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    values = frame.loc[:, list(_COLUMNS)].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values")
    time = values[:, 0]
    dt = np.diff(time).astype(np.float32)
    if np.any(dt <= 0):
        raise ValueError(f"{path} has non-positive time increments")
    rotation = quat_to_matrix(values[:, 4:8]).astype(np.float32)
    return PreparedFlight(
        name=path.stem,
        position=values[:, 1:4],
        rotation=rotation,
        velocity_world=values[:, 8:11],
        omega_body=values[:, 11:14],
        pwm=values[:-1, 14:18],
        voltage=np.zeros(len(dt), dtype=np.float32),
        dt=dt,
        attitude="motion capture",
    )


def load_idsia_prepared(
    root: str | Path | None = None,
) -> tuple[list[PreparedFlight], list[PreparedFlight], list[PreparedFlight]]:
    """Runs 1–3 train, run 4 validates, and all Melon runs test."""
    root = DEFAULT_ROOT if root is None else Path(root)
    train_dir = root / "data" / "train"
    test_dir = root / "data" / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"IDSIA benchmark data not found under {root}; clone "
            "https://github.com/idsia-robotics/nanodrone-sysid-benchmark "
            f"into {DEFAULT_ROOT}"
        )

    train_paths = [
        train_dir / f"{trajectory}_20251017_run{run}.csv"
        for trajectory in TRAIN_TRAJECTORIES
        for run in (1, 2, 3)
    ]
    validation_paths = [
        train_dir / f"{trajectory}_20251017_run4.csv"
        for trajectory in TRAIN_TRAJECTORIES
    ]
    test_paths = sorted(test_dir.glob("melon_*.csv"))
    missing = [path for path in train_paths + validation_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing IDSIA benchmark files: {missing}")
    if not test_paths:
        raise FileNotFoundError(f"no Melon test flights under {test_dir}")
    return (
        [_load(path) for path in train_paths],
        [_load(path) for path in validation_paths],
        [_load(path) for path in test_paths],
    )
