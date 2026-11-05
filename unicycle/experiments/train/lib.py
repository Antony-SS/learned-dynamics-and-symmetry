"""Open-loop window trainer. Best checkpoint is lowest val xyyaw at ``HORIZON_S``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from quadrotor.geometry import train_device
from unicycle.experiments.data import Trajectory, load_split
from unicycle.experiments.models import QUOTIENT_AUG, load_state, make_model
from unicycle.experiments.utils.params import (
    BATCH,
    EPOCHS,
    HEADING_LENGTH,
    HORIZON_S,
    LR,
    PATIENCE,
    RUNS,
    n_params,
    run_dir,
)
from unicycle.symmetry import wrap_angle


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _steps(horizon_s: float, dt: float) -> int:
    return max(int(round(horizon_s / dt)), 1)


def windows(logs: list[Trajectory], horizon_s: float, stride_s: float | None = None):
    """Stack start pose, start rates, command sequences, and target poses."""
    poses0, rates0, commands, targets, dts = [], [], [], [], []
    for log in logs:
        h = _steps(horizon_s, log.dt)
        stride = 1 if stride_s is None else max(int(round(stride_s / log.dt)), 1)
        if log.n_steps < h:
            continue
        for i in range(0, log.n_steps - h + 1, stride):
            poses0.append(log.poses[i])
            rates0.append(log.rates[i])
            commands.append(log.commands[i : i + h])
            targets.append(log.poses[i + 1 : i + 1 + h])
            dts.append(log.dt)
    if not poses0:
        raise ValueError(f"no windows of {horizon_s:g} s")
    return (
        torch.tensor(np.stack(poses0), dtype=torch.float32),
        torch.tensor(np.stack(rates0), dtype=torch.float32),
        torch.tensor(np.stack(commands), dtype=torch.float32),
        torch.tensor(np.stack(targets), dtype=torch.float32),
        torch.tensor(np.asarray(dts), dtype=torch.float32),
    )


def xyyaw_loss(pred: torch.Tensor, target: torch.Tensor, heading_length: float = HEADING_LENGTH) -> torch.Tensor:
    pos = ((pred[..., :2] - target[..., :2]) ** 2).sum(dim=-1)
    yaw = (heading_length * wrap_angle(pred[..., 2] - target[..., 2])) ** 2
    return (pos + yaw).mean()


def feature_stats(family: str, logs: list[Trajectory]) -> tuple[torch.Tensor, torch.Tensor]:
    model = make_model(family)
    feats = []
    with torch.no_grad():
        for log in logs:
            pose = torch.tensor(log.poses[:-1], dtype=torch.float32)
            rates = torch.tensor(log.rates, dtype=torch.float32)
            command = torch.tensor(log.commands, dtype=torch.float32)
            feats.append(model.feature_map(pose, rates, command))
    stacked = torch.cat(feats, dim=0)
    mean = stacked.mean(dim=0)
    scale = stacked.std(dim=0).clamp_min(1e-6)
    return mean, scale


def _configure(family: str, train: list[Trajectory], device: torch.device) -> torch.nn.Module:
    model = make_model(family).to(device)
    mean, scale = feature_stats(family, train)
    model.set_feature_stats(mean.to(device), scale.to(device))
    return model


def _split_loss(model, pose, rates, command, target, dt, device) -> float:
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for start in range(0, len(pose), BATCH):
            pred = model.rollout(
                pose[start : start + BATCH].to(device),
                rates[start : start + BATCH].to(device),
                command[start : start + BATCH].to(device),
                float(dt[start]),
            )
            total += float(xyyaw_loss(pred, target[start : start + BATCH].to(device)))
            batches += 1
    return total / max(batches, 1)


def train_family(
    family: str,
    seed: int = 0,
    *,
    device: torch.device | None = None,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
) -> torch.nn.Module:
    device = train_device() if device is None else device
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    folder = run_dir(family, seed)
    folder.mkdir(parents=True, exist_ok=True)
    last_path = folder / "last.pt"
    best_path = folder / "best.pt"
    train, val, _test = load_split(augment_train=family == QUOTIENT_AUG)
    model = _configure(family, train, device)
    train_p, train_r, train_u, train_y, train_dt = windows(train, HORIZON_S)
    val_p, val_r, val_u, val_y, val_dt = windows(val, HORIZON_S)
    opt = torch.optim.Adam(model.neural.parameters(), lr=LR)
    start_epoch = 1
    curve: list[dict] = []
    best_val = float("inf")
    stale = 0
    if last_path.is_file():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        load_state(model, payload)
        opt.load_state_dict(payload["optimizer"])
        curve = list(payload.get("curve", []))
        best_val = float(payload.get("best_val", float("inf")))
        stale = int(payload.get("stale", 0))
        start_epoch = int(payload["epoch"]) + 1
        print(f"{family}  seed {seed}  resume epoch {start_epoch}  val {best_val:.6f}", flush=True)
    else:
        print(
            f"{family}  seed {seed}  params {n_params(model)}  "
            f"windows {len(train_p)}  {HORIZON_S:g} s train / {HORIZON_S:g} s val xyyaw",
            flush=True,
        )

    n = len(train_p)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        order = torch.randperm(n)
        train_loss = 0.0
        batches = 0
        for start in range(0, n, BATCH):
            index = order[start : start + BATCH]
            pred = model.rollout(
                train_p[index].to(device),
                train_r[index].to(device),
                train_u[index].to(device),
                float(train_dt[int(index[0])]),
            )
            loss = xyyaw_loss(pred, train_y[index].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.detach())
            batches += 1
        train_mean = train_loss / max(batches, 1)
        val_mean = _split_loss(model, val_p, val_r, val_u, val_y, val_dt, device)
        mark = ""
        if val_mean < best_val:
            best_val = val_mean
            stale = 0
            torch.save({"state": snapshot(model), "epoch": epoch, "val": best_val, "seed": seed}, best_path)
            mark = "  saved"
        else:
            stale += 1
        curve.append({"epoch": epoch, "train": train_mean, "val": val_mean})
        torch.save(
            {
                "state": snapshot(model),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "curve": curve,
                "best_val": best_val,
                "stale": stale,
                "seed": seed,
                "family": family,
            },
            last_path,
        )
        write_json(folder / "curve.json", {"curve": curve, "best_val": best_val})
        print(f"  {family}  seed {seed}  epoch {epoch:02d}  train {train_mean:.6f}  val {val_mean:.6f}{mark}", flush=True)
        if patience > 0 and stale >= patience:
            print(
                f"  {family}  seed {seed}  stop after {epoch} epochs  "
                f"no val improvement for {patience}",
                flush=True,
            )
            break

    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    load_state(model, payload)
    model.eval()
    return model


def load_trained(family: str, seed: int, device: torch.device | None = None) -> torch.nn.Module:
    device = train_device() if device is None else device
    path = run_dir(family, seed) / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    train, _val, _test = load_split(augment_train=family == QUOTIENT_AUG)
    model = _configure(family, train, device)
    load_state(model, payload)
    model.to(device)
    model.eval()
    return model


def list_seeds(family: str) -> list[int]:
    folder = RUNS / family
    if not folder.is_dir():
        return []
    seeds = []
    for child in sorted(folder.glob("seed_*")):
        if (child / "best.pt").is_file():
            seeds.append(int(child.name.split("_")[1]))
    return seeds
