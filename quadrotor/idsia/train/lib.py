"""Shared IDSIA trainer: their recipe, resume last.pt, keep best.pt."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from quadrotor.geometry import train_device
from quadrotor.idsia.baselines.published import _windows, fit_canonical_scalers, fit_scalers
from quadrotor.idsia.data.load import load_idsia_prepared
from quadrotor.idsia.baselines.published import PhysQuadModel, PhysResQuadModel, ResidualQuadModel
from quadrotor.idsia.train.models import make_model
from quadrotor.idsia.utils.idsia_training_reproduction_params import (
    BATCH,
    EPOCHS,
    HORIZON,
    PHYS_RES,
    LR_END,
    LR_START,
    SCHED_T,
    STRIDE,
    n_params,
    run_dir,
    run_roots,
    scaled_loss,
    time_weights,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _feature_stats(_family: str, flights):
    return fit_canonical_scalers(flights)


def _prepare(family: str, train, val):
    x_mean, x_scale, _u_mean, _u_scale = fit_scalers(train)
    feat_mean, feat_scale = _feature_stats(family, train)
    train_x, train_u, train_y = _windows(train, HORIZON, stride=STRIDE)
    val_x, val_u, val_y = _windows(val, HORIZON, stride=STRIDE)
    return {
        "x_scale": x_scale,
        "feat_mean": feat_mean,
        "feat_scale": feat_scale,
        "train_x": train_x,
        "train_u": train_u,
        "train_y": train_y,
        "val_x": val_x,
        "val_u": val_u,
        "val_y": val_y,
    }


def _configure(model, stats, device):
    model = model.to(device)
    model.set_feature_stats(stats["feat_mean"].to(device), stats["feat_scale"].to(device))
    return model


def _split_loss(model, x, u, y, scale, weights, device) -> float:
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for start in range(0, len(x), BATCH):
            pred = model(x[start : start + BATCH].to(device), u[start : start + BATCH].to(device))
            total += float(scaled_loss(pred, y[start : start + BATCH].to(device), scale, weights))
            batches += 1
    return total / max(batches, 1)


def train_family(
    family: str,
    seed: int = 0,
    *,
    device: torch.device | None = None,
    epochs: int = EPOCHS,
    symmetric: bool = True,
) -> torch.nn.Module:
    device = train_device() if device is None else device
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    folder = run_dir(family, seed)
    folder.mkdir(parents=True, exist_ok=True)
    last_path = folder / "last.pt"
    best_path = folder / "best.pt"
    train, val, _test = load_idsia_prepared()
    weights = time_weights(device)
    stats = _prepare(family, train, val)
    model = _configure(make_model(family, symmetric=symmetric), stats, device)
    x_scale = stats["x_scale"].to(device).view(1, 1, -1)
    train_x, train_u, train_y = stats["train_x"], stats["train_u"], stats["train_y"]
    val_x, val_u, val_y = stats["val_x"], stats["val_u"], stats["val_y"]
    opt = torch.optim.Adam(model.neural.parameters(), lr=LR_START)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SCHED_T, eta_min=LR_END)
    start_epoch = 1
    curve: list[dict] = []
    best_val = float("inf")
    if last_path.is_file():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(adapt_state(payload["state"], model), strict=False)
        opt.load_state_dict(payload["optimizer"])
        sched.load_state_dict(payload["scheduler"])
        curve = list(payload.get("curve", []))
        best_val = float(payload.get("best_val", float("inf")))
        start_epoch = int(payload["epoch"]) + 1
        print(f"{family}  seed {seed}  resume epoch {start_epoch}  val {best_val:.6f}", flush=True)
    else:
        print(
            f"{family}  seed {seed}  params {n_params(model)}  windows {len(train_x)}  "
            f"stride {STRIDE}  Adam {LR_START:g}→{LR_END:g}  WeightedMSE",
            flush=True,
        )

    n = len(train_x)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        order = torch.randperm(n)
        train_loss = 0.0
        batches = 0
        for start in range(0, n, BATCH):
            index = order[start : start + BATCH]
            pred = model(train_x[index].to(device), train_u[index].to(device))
            loss = scaled_loss(pred, train_y[index].to(device), x_scale, weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.detach())
            batches += 1
        sched.step()
        train_mean = train_loss / max(batches, 1)
        val_mean = _split_loss(model, val_x, val_u, val_y, x_scale, weights, device)
        mark = ""
        if val_mean < best_val:
            best_val = val_mean
            torch.save({"state": snapshot(model), "epoch": epoch, "val": best_val, "seed": seed}, best_path)
            mark = "  saved"
        curve.append({"epoch": epoch, "train": train_mean, "val": val_mean})
        torch.save(
            {
                "state": snapshot(model),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "epoch": epoch,
                "curve": curve,
                "best_val": best_val,
                "seed": seed,
                "family": family,
            },
            last_path,
        )
        write_json(folder / "curve.json", {"curve": curve, "best_val": best_val})
        print(f"  {family}  seed {seed}  epoch {epoch:02d}  train {train_mean:.6f}  val {val_mean:.6f}{mark}", flush=True)

    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(adapt_state(payload["state"], model), strict=False)
    model.eval()
    return model


def _load_phys_res(seed: int, device: torch.device) -> torch.nn.Module:
    path = run_dir(PHYS_RES, seed) / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = PhysResQuadModel(PhysQuadModel(zero_omega_dot=True), ResidualQuadModel(hidden_dim=64, num_layers=5))
    model.load_state_dict(payload["state"], strict=False)
    model.to(device)
    model.eval()
    return model


def adapt_state(state: dict, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Map older checkpoint names onto the current module tree.

    Early ``phys_quotient_sym`` saved the orbit MLP as ``neural.mlp`` / ``neural.out``
    and scalers as ``canonical_*``. ``strict=False`` then left a zero leftover,
    which is just physics.
    """
    names = set(model.state_dict())
    adapted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        dest = key
        if dest == "canonical_mean":
            dest = "feat_mean"
        elif dest == "canonical_scale":
            dest = "feat_scale"
        elif dest.startswith("neural.mlp") or dest.startswith("neural.out"):
            nested = "neural.residual." + dest[len("neural.") :]
            if nested in names:
                dest = nested
        if dest in names:
            adapted[dest] = value
    leftover = [name for name in names if name.endswith("out.weight")]
    if leftover and not any(name in adapted for name in leftover):
        raise RuntimeError(f"checkpoint leftover keys did not match {leftover}")
    return adapted


def load_trained(family: str, seed: int, device: torch.device | None = None, symmetric: bool = True) -> torch.nn.Module:
    device = train_device() if device is None else device
    if family == PHYS_RES:
        return _load_phys_res(seed, device)
    path = run_dir(family, seed) / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    train, val, _test = load_idsia_prepared()
    stats = _prepare(family, train, val)
    model = _configure(make_model(family, symmetric=symmetric), stats, device)
    model.load_state_dict(adapt_state(payload["state"], model), strict=False)
    model.to(device)
    model.eval()
    return model


def list_seeds(family: str) -> list[int]:
    seeds: set[int] = set()
    for root in run_roots():
        folder = root / family
        if not folder.is_dir():
            continue
        for child in folder.glob("seed_*"):
            if (child / "best.pt").is_file():
                seeds.add(int(child.name.split("_")[1]))
    return sorted(seeds)
