"""Melon MAE and saved predictions at h=1, 10, 50."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from quadrotor.evaluate import benchmark_mae
from quadrotor.idsia.models import QuadDynamicsNet, quotient
from quadrotor.idsia.recipe import BATCH, EVAL, HORIZON, HS, KEYS, eval_dir
from quadrotor.idsia.train_lib import pack_dynamics, write_json
from quadrotor.train import _apply, _so3_log, module_device


def pack_mae(mae: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {k: [float(mae[k][i]) for i in HS] for k in KEYS}


def _dynamics_mae(model: QuadDynamicsNet, flights, device) -> tuple[dict[str, np.ndarray], dict]:
    packed = pack_dynamics(flights, device, model.history, HORIZON, stride=1)
    n = packed["p"].shape[0]
    totals = {k: np.zeros(HORIZON) for k in KEYS}
    snaps = {k: [] for k in KEYS}
    model.eval()
    with torch.no_grad():
        for start in range(0, n, BATCH):
            sl = slice(start, min(start + BATCH, n))
            p = packed["p"][sl]
            r = packed["r"][sl]
            v = packed["v"][sl]
            w = packed["w"][sl]
            pwm = packed["pwm"][sl]
            dt = packed["dt"][sl]
            buf = packed["hist"][sl].clone()
            y = packed["y"][sl]
            rt = packed["rt"][sl]
            err = {k: [] for k in KEYS}
            for k in range(HORIZON):
                flat = ((buf - model.x_mean) / model.x_scale).reshape(buf.shape[0], -1)
                acc = model(flat)
                p, r, v, w = _apply(p, r, v, w, acc, dt[:, k])
                rel = torch.matmul(rt[:, k].transpose(-1, -2), r)
                err["position"].append(torch.linalg.vector_norm(p - y[:, k, 0:3], dim=-1))
                err["rotation"].append(torch.linalg.vector_norm(_so3_log(rel), dim=-1))
                err["velocity"].append(torch.linalg.vector_norm(v - y[:, k, 3:6], dim=-1))
                err["omega"].append(torch.linalg.vector_norm(w - y[:, k, 9:12], dim=-1))
                if k + 1 < HORIZON:
                    nxt = quotient(r, v, w, pwm[:, k + 1]).unsqueeze(1)
                    buf = torch.cat((buf[:, 1:], nxt), dim=1)
            for key in KEYS:
                stacked = torch.stack(err[key], dim=1)
                totals[key] += stacked.sum(0).cpu().numpy()
                snaps[key].append(stacked[:, list(HS)].cpu().numpy())
    mae = {k: totals[k] / max(n, 1) for k in KEYS}
    preds = {k: np.concatenate(snaps[k], axis=0) for k in KEYS}
    return mae, preds


def melon_mae(model, flights) -> dict[str, np.ndarray]:
    if isinstance(model, QuadDynamicsNet):
        mae, _ = _dynamics_mae(model, flights, module_device(model))
        return mae
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
    if isinstance(model, QuadDynamicsNet):
        mae, preds = _dynamics_mae(model, flights, module_device(model))
        save_eval(family, seed, mae, preds)
    else:
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
