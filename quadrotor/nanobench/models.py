"""NanoBench symmetric leftover trained on the A1b multisine flights only."""

from __future__ import annotations

from pathlib import Path

import torch

from quadrotor.canonical import load_prepared
from quadrotor.fit import load_or_train_symmetric

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "runs" / "nanobench_parity18k"


def load_or_train(
    folder: str | Path | None = None,
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
    epochs: int = 20,
):
    train, val, test = load_prepared(data_rich=False)
    model, history = load_or_train_symmetric(
        train,
        val,
        test,
        CHECKPOINT_DIR if folder is None else folder,
        seed=seed,
        device=device,
        epochs=epochs,
        stride=1,
        horizon=50,
        include_voltage=False,
        name="symmetric",
    )
    return (train, val, test), model, history
