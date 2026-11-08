"""Unicycle experiment constants."""

from __future__ import annotations

from pathlib import Path

from unicycle.experiments.data import MAX_LINEAR_VELOCITY
from unicycle.experiments.models import NAIVE, PHYSICS, QUOTIENT, QUOTIENT_AUG, QUOTIENT_SYM

HORIZON_S = 10.0
TEST_HORIZONS_S = (1.0, 10.0, 30.0)
ROLLOUT_VIZ_WINDOW_S = 30.0
ROLLOUT_VIZ_STARTS_S = (0.0, 30.0, 60.0)
HEADING_LENGTH = MAX_LINEAR_VELOCITY
EPOCHS = 40
BATCH = 256
LR = 1e-3  # cosine-annealed to LR_MIN over EPOCHS
LR_MIN = 1e-5
SEEDS = (0, 1, 2, 3, 4)

RUNS = Path(__file__).resolve().parents[3] / "runs" / "unicycle"
EVAL = RUNS / "eval"
VIZ = Path(__file__).resolve().parents[1] / "visualizations"

FAMILIES = (PHYSICS, NAIVE, QUOTIENT, QUOTIENT_AUG, QUOTIENT_SYM)
TRAIN_FAMILIES = (NAIVE, QUOTIENT, QUOTIENT_AUG, QUOTIENT_SYM)


def run_dir(family: str, seed: int) -> Path:
    return RUNS / family / f"seed_{int(seed)}"


def eval_dir(family: str, seed: int) -> Path:
    return EVAL / family / f"seed_{int(seed)}"


def n_params(model) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
