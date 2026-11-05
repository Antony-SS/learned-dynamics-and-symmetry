"""Unicycle experiment constants."""

from __future__ import annotations

from pathlib import Path

from unicycle.experiments.data import MAX_LINEAR_VELOCITY
from unicycle.experiments.models import NAIVE, PHYSICS, QUOTIENT, QUOTIENT_AUG, QUOTIENT_SYM

HORIZON_S = 5.0
TEST_HORIZONS_S = (1.0, 5.0, 30.0)
HEADING_LENGTH = MAX_LINEAR_VELOCITY
EPOCHS = 40
BATCH = 64
LR = 1e-4
SEEDS = (0, 1, 2, 3, 4)

OUT = Path(__file__).resolve().parents[1] / "out"
RUNS = OUT / "runs"
EVAL = OUT / "eval"
VIZ = Path(__file__).resolve().parents[1] / "visualizations"

FAMILIES = (PHYSICS, NAIVE, QUOTIENT, QUOTIENT_AUG, QUOTIENT_SYM)
TRAIN_FAMILIES = (NAIVE, QUOTIENT, QUOTIENT_AUG, QUOTIENT_SYM)


def run_dir(family: str, seed: int) -> Path:
    return RUNS / family / f"seed_{int(seed)}"


def eval_dir(family: str, seed: int) -> Path:
    return EVAL / family / f"seed_{int(seed)}"


def n_params(model) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
