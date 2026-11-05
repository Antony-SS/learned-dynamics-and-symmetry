"""Test-set xyyaw error at 1 s, 5 s, 30 s, and the full collection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from quadrotor.geometry import train_device
from unicycle.experiments.data import Trajectory, load_split
from unicycle.experiments.models import PHYSICS, make_model
from unicycle.experiments.train.lib import write_json, xyyaw_loss
from unicycle.experiments.utils.params import EVAL, TEST_HORIZONS_S, eval_dir
from unicycle.symmetry import wrap_angle


def _horizon_errors(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    dxy = pred[..., :2] - target[..., :2]
    pos = float(torch.sqrt((dxy ** 2).sum(dim=-1).mean()))
    heading = float(torch.sqrt((wrap_angle(pred[..., 2] - target[..., 2]) ** 2).mean()))
    return {"position_rmse": pos, "heading_rmse": heading, "xyyaw": float(xyyaw_loss(pred, target))}


def _windows(logs: list[Trajectory], horizon_s: float):
    poses0, rates0, commands, targets = [], [], [], []
    dt = logs[0].dt
    h = max(int(round(horizon_s / dt)), 1)
    for log in logs:
        if log.n_steps < h:
            continue
        for i in range(0, log.n_steps - h + 1):
            poses0.append(log.poses[i])
            rates0.append(log.rates[i])
            commands.append(log.commands[i : i + h])
            targets.append(log.poses[i + 1 : i + 1 + h])
    return (
        torch.tensor(np.stack(poses0), dtype=torch.float32),
        torch.tensor(np.stack(rates0), dtype=torch.float32),
        torch.tensor(np.stack(commands), dtype=torch.float32),
        torch.tensor(np.stack(targets), dtype=torch.float32),
        dt,
        h,
    )


@torch.no_grad()
def eval_horizon(model, logs: list[Trajectory], horizon_s: float, device: torch.device) -> dict[str, float]:
    pose, rates, command, target, dt, _h = _windows(logs, horizon_s)
    preds = []
    for start in range(0, len(pose), 64):
        preds.append(
            model.rollout(
                pose[start : start + 64].to(device),
                rates[start : start + 64].to(device),
                command[start : start + 64].to(device),
                dt,
            ).cpu()
        )
    return _horizon_errors(torch.cat(preds, dim=0), target)


@torch.no_grad()
def eval_full(model, logs: list[Trajectory], device: torch.device) -> dict[str, float]:
    pos, heading, xyyaw = [], [], []
    for log in logs:
        pose = torch.tensor(log.poses[0], dtype=torch.float32, device=device).unsqueeze(0)
        rates = torch.tensor(log.rates[0], dtype=torch.float32, device=device).unsqueeze(0)
        command = torch.tensor(log.commands, dtype=torch.float32, device=device).unsqueeze(0)
        target = torch.tensor(log.poses[1:], dtype=torch.float32)
        pred = model.rollout(pose, rates, command, log.dt).cpu().squeeze(0)
        row = _horizon_errors(pred, target)
        pos.append(row["position_rmse"])
        heading.append(row["heading_rmse"])
        xyyaw.append(row["xyyaw"])
    return {
        "position_rmse": float(np.mean(pos)),
        "heading_rmse": float(np.mean(heading)),
        "xyyaw": float(np.mean(xyyaw)),
    }


def eval_model(model, logs: list[Trajectory], family: str, seed: int, device: torch.device | None = None) -> dict:
    device = train_device() if device is None else device
    model.eval()
    payload = {"family": family, "seed": seed}
    for horizon in TEST_HORIZONS_S:
        payload[f"{horizon:g}s"] = eval_horizon(model, logs, horizon, device)
    payload["full"] = eval_full(model, logs, device)
    dest = eval_dir(family, seed)
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "test.json", payload)
    return payload


def family_summary(family: str) -> dict:
    rows = []
    folder = EVAL / family
    for path in sorted(folder.glob("seed_*/test.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        raise FileNotFoundError(f"no eval under {folder}")
    keys = [f"{h:g}s" for h in TEST_HORIZONS_S] + ["full"]
    summary = {"family": family, "n": len(rows)}
    for key in keys:
        for metric in ("position_rmse", "heading_rmse", "xyyaw"):
            values = [row[key][metric] for row in rows]
            summary.setdefault(key, {})[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    write_json(folder / "summary.json", summary)
    return summary


def load_physics(device: torch.device | None = None) -> torch.nn.Module:
    device = train_device() if device is None else device
    model = make_model(PHYSICS).to(device)
    model.eval()
    return model


def test_logs():
    _train, _val, test = load_split()
    return test
