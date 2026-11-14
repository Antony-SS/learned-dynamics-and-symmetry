"""Busetto et al. training recipe, in one place.

Stride 1, 50-step windows, WeightedMSE λ=0.1, Adam 1e-5→1e-8 cosine T=500,
20 epochs, official Square/Random/Chirp 1–3 / run 4 / Melon split.
"""

from __future__ import annotations

from pathlib import Path

import torch

HORIZON = 50
STRIDE = 1
EPOCHS = 20
BATCH = 256
LR_START = 1e-5
LR_END = 1e-8
SCHED_T = 500
WEIGHT_LAMBDA = 0.1
HS = (0, 9, 49)
KEYS = ("position", "rotation", "velocity", "omega")

OUT = Path(__file__).resolve().parents[1] / "out"
# Existing trainings live at repo ``runs/quadrotor``. Also accept ``idsia/out/runs``.
_REPO_RUNS = Path(__file__).resolve().parents[3] / "runs" / "quadrotor"
RUNS = _REPO_RUNS
EVAL = OUT / "eval"


def run_roots() -> tuple[Path, ...]:
    return (_REPO_RUNS, OUT / "runs")


def run_dir(family: str, seed: int) -> Path:
    for root in run_roots():
        folder = root / family / f"seed_{int(seed)}"
        if (folder / "best.pt").is_file() or (folder / "last.pt").is_file():
            return folder
    return _REPO_RUNS / family / f"seed_{int(seed)}"

# Loaded baselines, never trained here: analytic physics and the authors'
# released physics+residual checkpoint (``baselines.load_phys_res``).
PHYSICS = "physics"
PHYS_RES = "phys_res"
BASELINES = (PHYSICS, PHYS_RES)
SYM_QUAD_DYNAMICS = "sym_quad_dynamics"
PHYS_QUOTIENT = "phys_quotient"
PHYS_QUOTIENT_SYM = "phys_quotient_sym"
# Same network as phys_quotient, trained on every flight plus its three Z₂×Z₂
# mirror images (quadrotor.symmetry.orbit_flights). Validation stays unmirrored.
PHYS_QUOTIENT_AUG = "phys_quotient_aug"
AUGMENTED_FAMILIES = (PHYS_QUOTIENT_AUG,)
# Their phys+res architecture (frozen RK4 + normalized-world MLP) retrained
# with the same recipe and seeds as our families. Separate name, so it never
# collides with the published checkpoint's eval under ``phys_res``.
PHYS_RES_RETRAIN = "phys_res_retrain"
TRAIN_FAMILIES = (PHYS_RES_RETRAIN, PHYS_QUOTIENT, PHYS_QUOTIENT_AUG, PHYS_QUOTIENT_SYM, SYM_QUAD_DYNAMICS)
FAMILIES = BASELINES + TRAIN_FAMILIES


def eval_dir(family: str, seed: int) -> Path:
    return EVAL / family / f"seed_{int(seed)}"


def time_weights(device: torch.device, horizon: int = HORIZON) -> torch.Tensor:
    weights = torch.exp(
        -WEIGHT_LAMBDA * torch.arange(horizon, device=device, dtype=torch.float32)
    )
    return weights / weights.sum()


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((pred - target).pow(2) * weights.view(1, -1, 1)).mean()


def scaled_loss(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return weighted_mse((pred - target) / scale, torch.zeros_like(pred), weights)


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
