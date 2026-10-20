"""Unicycle inputs and labels.

The input is the wheel step ``(v·dt, ω·dt)``, which is unchanged by an SE(2)
motion of the arena. The label is the full body step ``Log(T_t^{-1} T_{t+1})``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from unicycle.data import Trajectory
from unicycle.geometry import body_steps

FEATURES = ("v_dt", "omega_dt")
LABELS = ("dx", "dy", "dtheta")


def feature_names() -> tuple[str, ...]:
    return FEATURES


def feature_dim() -> int:
    return len(FEATURES)


def label_dim() -> int:
    return len(LABELS)


def features(log: Trajectory) -> np.ndarray:
    """One row per step. Shape ``(N-1, 2)``."""
    return np.asarray(log.increments, dtype=np.float64)


def labels(log: Trajectory) -> np.ndarray:
    return body_steps(log.poses)


def collect(logs: Sequence[Trajectory]) -> tuple[np.ndarray, np.ndarray]:
    xs = [features(log) for log in logs]
    ys = [labels(log) for log in logs]
    return (
        np.concatenate(xs, axis=0).astype(np.float32),
        np.concatenate(ys, axis=0).astype(np.float32),
    )
