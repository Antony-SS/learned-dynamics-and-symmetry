"""Compatibility re-export. IDSIA data lives in ``quadrotor.idsia.data``."""

from quadrotor.idsia.data import (
    DEFAULT_ROOT,
    TRAIN_TRAJECTORIES,
    load_idsia_prepared,
)

__all__ = ["DEFAULT_ROOT", "TRAIN_TRAJECTORIES", "load_idsia_prepared"]
