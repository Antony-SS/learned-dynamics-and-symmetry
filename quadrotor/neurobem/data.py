"""NeuroBEM processed-data loader for recurrent state rollouts.

Native 400 Hz. A 200-step window is 0.50 s. Motor columns are reordered into
front-right, back-right, back-left, front-left, matching the Crazyflie-X
convention used by the symmetry implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quadrotor.canonical import PreparedFlight
from quadrotor.geometry import quat_to_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "neurobem"
RATE_HZ = 400
HORIZON_STEPS = 200
DOWNSAMPLE = 1


def _segment_id(path: Path) -> str:
    return path.stem.removeprefix("merged_")


def _flight_id(path: Path) -> str:
    return _segment_id(path).split("_seg_", 1)[0]


def _load(path: Path) -> PreparedFlight:
    frame = pd.read_csv(path).iloc[::DOWNSAMPLE].reset_index(drop=True)
    quaternion = frame[["quat x", "quat y", "quat z", "quat w"]].to_numpy(
        dtype=np.float32
    )
    rotation = quat_to_matrix(quaternion).astype(np.float32)
    velocity_body = frame[["vel x", "vel y", "vel z"]].to_numpy(dtype=np.float32)
    velocity_world = np.einsum("nij,nj->ni", rotation, velocity_body).astype(np.float32)
    # Source order: back-right, front-right, back-left, front-left.
    motors = frame[["mot 2", "mot 1", "mot 3", "mot 4"]].to_numpy(dtype=np.float32)
    time = frame["t"].to_numpy(dtype=np.float64)
    dt = np.diff(time).astype(np.float32)
    if np.any(dt <= 0):
        raise ValueError(f"{path} has non-positive time increments")
    return PreparedFlight(
        name=_segment_id(path),
        position=frame[["pos x", "pos y", "pos z"]].to_numpy(dtype=np.float32),
        rotation=rotation,
        velocity_world=velocity_world,
        omega_body=frame[["ang vel x", "ang vel y", "ang vel z"]].to_numpy(
            dtype=np.float32
        ),
        motors=motors[:-1],
        dt=dt,
        attitude="Vicon",
    )


def load_neurobem_prepared(
    root: str | Path | None = None,
    *,
    seed: int = 0,
) -> tuple[list[PreparedFlight], list[PreparedFlight], list[PreparedFlight]]:
    """Official test segments plus a deterministic flight-level validation set."""
    root = DEFAULT_ROOT if root is None else Path(root)
    data_dir = root / "processed_data"
    test_file = root / "testset.txt"
    paths = sorted(data_dir.glob("merged_*.csv"))
    if not paths or not test_file.is_file():
        raise FileNotFoundError(
            f"NeuroBEM data not found under {root}; download processed_data.zip "
            "and testset.txt from https://download.ifi.uzh.ch/rpg/NeuroBEM/"
        )
    test_ids = {
        line.strip() for line in test_file.read_text().splitlines() if line.strip()
    }
    test_paths = [path for path in paths if _segment_id(path) in test_ids]
    remaining = [path for path in paths if _segment_id(path) not in test_ids]
    if len(test_paths) != len(test_ids):
        found = {_segment_id(path) for path in test_paths}
        raise FileNotFoundError(f"missing official test segments: {sorted(test_ids - found)}")

    # Keep every segment from one recording in one split. Selecting 20% of the
    # non-test flights approximates the paper's 70/20/10 partition while the
    # released testset.txt fixes the scientifically important test subset.
    by_flight: dict[str, list[Path]] = {}
    for path in remaining:
        by_flight.setdefault(_flight_id(path), []).append(path)
    flight_ids = sorted(by_flight)
    rng = np.random.default_rng(seed)
    shuffled = [flight_ids[index] for index in rng.permutation(len(flight_ids))]
    n_validation = max(1, int(round(0.20 * len(shuffled))))
    validation_ids = set(shuffled[:n_validation])
    validation_paths = [
        path for flight in validation_ids for path in sorted(by_flight[flight])
    ]
    train_paths = [
        path
        for flight in shuffled[n_validation:]
        for path in sorted(by_flight[flight])
    ]
    return (
        [_load(path) for path in train_paths],
        [_load(path) for path in validation_paths],
        [_load(path) for path in test_paths],
    )
