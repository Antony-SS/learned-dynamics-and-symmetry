"""NeuroBEM symmetric leftover: ParityMotorNet-24 at native 400 Hz, h=200."""

from __future__ import annotations

from pathlib import Path

import torch

from quadrotor.fit import load_or_train_symmetric
from quadrotor.neurobem.data import HORIZON_STEPS, load_neurobem_prepared

CHECKPOINT_DIR = Path(__file__).resolve().parent / "out" / "parity18k"
# 400 Hz: stride 40 is 0.10 s between windows, same spacing as stride 10 at 100 Hz.
STRIDE = 40


def load_or_train(
    folder: str | Path | None = None,
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
    epochs: int = 20,
):
    train, val, test = load_neurobem_prepared(seed=0)
    model, history = load_or_train_symmetric(
        train,
        val,
        test,
        CHECKPOINT_DIR if folder is None else folder,
        seed=seed,
        device=device,
        epochs=epochs,
        stride=STRIDE,
        horizon=HORIZON_STEPS,
        warm_start=True,
        warm_steps=20,
        include_voltage=False,
        name="symmetric",
    )
    return (train, val, test), model, history
