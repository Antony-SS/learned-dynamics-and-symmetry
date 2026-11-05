"""Melon MAE and saved predictions at h=1, 10, 50."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quadrotor.evaluate import benchmark_mae
from quadrotor.idsia.train.lib import write_json
from quadrotor.idsia.utils.idsia_training_reproduction_params import EVAL, HORIZON, HS, KEYS, eval_dir


def pack_mae(mae: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {k: [float(mae[k][i]) for i in HS] for k in KEYS}


def melon_mae(model, flights) -> dict[str, np.ndarray]:
    return benchmark_mae(model, flights, horizon=HORIZON, start_stride=1)


def save_eval(
    family: str,
    seed: int,
    mae: dict[str, np.ndarray],
    predictions: dict | None = None,
) -> Path:
    folder = eval_dir(family, seed)
    folder.mkdir(parents=True, exist_ok=True)
    packed = pack_mae(mae)
    write_json(
        folder / "mae.json",
        {"family": family, "seed": int(seed), "h": [1, 10, 50], **{k: packed[k] for k in KEYS}},
    )
    if predictions is not None:
        np.savez(folder / "predictions.npz", **{k: np.asarray(v) for k, v in predictions.items()})
    return folder


def eval_model(model, flights, family: str, seed: int) -> dict[str, list[float]]:
    mae = benchmark_mae(model, flights, horizon=HORIZON, start_stride=1)
    save_eval(family, seed, mae, {k: np.asarray(mae[k])[list(HS)] for k in KEYS})
    return pack_mae(mae)


def family_summary(family: str) -> dict:
    root = EVAL / family
    rows = []
    if root.is_dir():
        for child in sorted(root.glob("seed_*")):
            path = child / "mae.json"
            if path.is_file():
                rows.append(json.loads(path.read_text()))
    summary: dict = {"family": family, "seeds": rows}
    if rows:
        for key in KEYS:
            arr = np.asarray([row[key] for row in rows], dtype=np.float64)
            summary[key] = {
                "mean": arr.mean(0).tolist(),
                "std": arr.std(0, ddof=1).tolist() if len(rows) > 1 else [0.0, 0.0, 0.0],
            }
    write_json(root / "summary.json", summary)
    return summary
