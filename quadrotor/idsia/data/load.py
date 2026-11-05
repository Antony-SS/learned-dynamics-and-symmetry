"""Official IDSIA nano-drone benchmark split and CSV conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quadrotor.flight import PreparedFlight

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "nanodrone-sysid-benchmark"
TRAIN_TRAJECTORIES = ("square", "random", "chirp")

def _quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """IDSIA CSVs store scalar-last quaternions; we keep rotations as SO(3)."""
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
    rotation = _quat_xyzw_to_matrix(values[:, 4:8]).astype(np.float32)
    return PreparedFlight(
        name=path.stem,
        position=values[:, 1:4],
        rotation=rotation,
        velocity_world=values[:, 8:11],
        omega_body=values[:, 11:14],
        motors=values[:-1, 14:18],  # m1_rads … m4_rads
        dt=dt,
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
