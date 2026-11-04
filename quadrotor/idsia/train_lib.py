"""Shared IDSIA trainer: their recipe, resume last.pt, keep best.pt."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from quadrotor.fit import train_device
from quadrotor.idsia.baselines import _windows, fit_canonical_scalers, fit_scalers
from quadrotor.idsia.data import load_idsia_prepared
from quadrotor.idsia.models import QuadDynamicsNet, make_model, quotient
from quadrotor.idsia.recipe import (
    BATCH,
    EPOCHS,
    HORIZON,
    LR_END,
    LR_START,
    SCHED_T,
    STRIDE,
    n_params,
    run_dir,
    scaled_loss,
    time_weights,
)
from quadrotor.train import _apply, _so3_log


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def flight_features(flight) -> np.ndarray:
    n = flight.n_steps
    rotation = flight.rotation[:n]
    up = np.array((0.0, 0.0, 1.0), np.float32)
    gravity = np.einsum("nji,j->ni", rotation, up)
    velocity = np.einsum("nji,nj->ni", rotation, flight.velocity_world[:n])
    return np.concatenate((gravity, velocity, flight.omega_body[:n], flight.pwm), axis=1).astype(
        np.float32
    )


def feat_stats(flights):
    rows = np.concatenate([flight_features(f) for f in flights], axis=0)
    return (
        torch.as_tensor(rows.mean(0), dtype=torch.float32),
        torch.as_tensor(np.clip(rows.std(0), 1e-6, None), dtype=torch.float32),
    )


def _starts(flights, horizon: int, stride: int):
    return [
        (i, s)
        for i, f in enumerate(flights)
        for s in range(0, f.n_steps - horizon, stride)
    ]


def pack_dynamics(flights, device: torch.device, history: int, horizon: int, stride: int):
    pairs = _starts(flights, horizon, stride)
    p0, r0, v0, w0, pwm, dt, pt, rt, vt, wt, hist, y = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for fi, s in pairs:
        f = flights[fi]
        n = f.n_steps
        feat = flight_features(f)
        idx = np.arange(s - history + 1, s + 1)
        hist.append(feat[np.clip(idx, 0, feat.shape[0] - 1)])
        stop = s + horizon
        p0.append(f.position[s])
        r0.append(f.rotation[s])
        v0.append(f.velocity_world[s])
        w0.append(f.omega_body[s])
        pwm.append(f.pwm[s:stop])
        dt.append(f.dt[s:stop])
        pt.append(f.position[s + 1 : stop + 1])
        rt.append(f.rotation[s + 1 : stop + 1])
        vt.append(f.velocity_world[s + 1 : stop + 1])
        wt.append(f.omega_body[s + 1 : stop + 1])
        so3 = _so3_log(torch.as_tensor(f.rotation[s + 1 : stop + 1])).numpy()
        y.append(
            np.concatenate(
                [
                    f.position[s + 1 : stop + 1],
                    f.velocity_world[s + 1 : stop + 1],
                    so3,
                    f.omega_body[s + 1 : stop + 1],
                ],
                axis=1,
            )
        )
    return {
        "p": torch.as_tensor(np.stack(p0), device=device),
        "r": torch.as_tensor(np.stack(r0), device=device),
        "v": torch.as_tensor(np.stack(v0), device=device),
        "w": torch.as_tensor(np.stack(w0), device=device),
        "pwm": torch.as_tensor(np.stack(pwm), device=device),
        "dt": torch.as_tensor(np.stack(dt), device=device),
        "hist": torch.as_tensor(np.stack(hist), device=device),
        "rt": torch.as_tensor(np.stack(rt), device=device),
        "y": torch.as_tensor(np.stack(y), device=device, dtype=torch.float32),
    }


def roll_dynamics(model: QuadDynamicsNet, packed, index) -> torch.Tensor:
    p = packed["p"][index]
    r = packed["r"][index]
    v = packed["v"][index]
    w = packed["w"][index]
    pwm = packed["pwm"][index]
    dt = packed["dt"][index]
    buf = packed["hist"][index].clone()
    preds = []
    for k in range(pwm.shape[1]):
        flat = ((buf - model.x_mean) / model.x_scale).reshape(buf.shape[0], -1)
        acc = model(flat)
        p, r, v, w = _apply(p, r, v, w, acc, dt[:, k])
        so3 = _so3_log(r)
        preds.append(torch.cat((p, v, so3, w), dim=-1))
        if k + 1 < pwm.shape[1]:
            nxt = quotient(r, v, w, pwm[:, k + 1]).unsqueeze(1)
            buf = torch.cat((buf[:, 1:], nxt), dim=1)
    return torch.stack(preds, 1)


def _split_loss_state(model, x, u, y, scale, weights, device) -> float:
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for start in range(0, len(x), BATCH):
            pred = model(x[start : start + BATCH].to(device), u[start : start + BATCH].to(device))
            total += float(scaled_loss(pred, y[start : start + BATCH].to(device), scale, weights))
            batches += 1
    return total / max(batches, 1)


def _split_loss_dynamics(model, packed, scale, weights) -> float:
    model.eval()
    n = packed["p"].shape[0]
    total = 0.0
    batches = 0
    with torch.no_grad():
        for start in range(0, n, BATCH):
            index = slice(start, start + BATCH)
            pred = roll_dynamics(model, packed, index)
            total += float(scaled_loss(pred, packed["y"][index], scale, weights))
            batches += 1
    return total / max(batches, 1)


def _prepare_state(family: str, train, val, device):
    x_mean, x_scale, u_mean, u_scale = fit_scalers(train)
    c_mean, c_scale = fit_canonical_scalers(train)
    train_x, train_u, train_y = _windows(train, HORIZON, stride=STRIDE)
    val_x, val_u, val_y = _windows(val, HORIZON, stride=STRIDE)
    if family == "phys_quotient":
        train_x = (train_x - x_mean) / x_scale
        train_u = (train_u - u_mean) / u_scale
        train_y = (train_y - x_mean) / x_scale
        val_x = (val_x - x_mean) / x_scale
        val_u = (val_u - u_mean) / u_scale
        val_y = (val_y - x_mean) / x_scale
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "u_mean": u_mean,
        "u_scale": u_scale,
        "c_mean": c_mean,
        "c_scale": c_scale,
        "train_x": train_x,
        "train_u": train_u,
        "train_y": train_y,
        "val_x": val_x,
        "val_u": val_u,
        "val_y": val_y,
    }


def _configure(model, family: str, stats, device):
    model = model.to(device)
    if family == "phys_quotient":
        with torch.no_grad():
            model.x_mean.copy_(stats["x_mean"])
            model.x_scale.copy_(stats["x_scale"])
            model.u_mean.copy_(stats["u_mean"])
            model.u_scale.copy_(stats["u_scale"])
            model.canonical_mean.copy_(stats["c_mean"])
            model.canonical_scale.copy_(stats["c_scale"])
        model.phys = model.phys.to(device)
    elif family == "phys_quotient_sym":
        with torch.no_grad():
            model.canonical_mean.copy_(stats["c_mean"])
            model.canonical_scale.copy_(stats["c_scale"])
        model.phys = model.phys.to(device)
    return model


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
    model = make_model(family, symmetric=symmetric)

    if family == "quad_dynamics":
        mean, scale = feat_stats(train)
        model = model.to(device)
        model.set_stats(mean.to(device), scale.to(device))
        packed = pack_dynamics(train, device, model.history, HORIZON, STRIDE)
        packed_val = pack_dynamics(val, device, model.history, HORIZON, STRIDE)
        x_scale = packed["y"].reshape(-1, 12).std(0).clamp_min(1e-6).to(device).view(1, 1, 12)
        opt = torch.optim.Adam(model.parameters(), lr=LR_START)
        trainable = model.parameters()
    else:
        stats = _prepare_state(family, train, val, device)
        model = _configure(model, family, stats, device)
        x_scale = stats["x_scale"].to(device).view(1, 1, -1)
        train_x, train_u, train_y = stats["train_x"], stats["train_u"], stats["train_y"]
        val_x, val_u, val_y = stats["val_x"], stats["val_u"], stats["val_y"]
        trainable = model.neural.parameters() if family == "phys_quotient_sym" else model.parameters()
        opt = torch.optim.Adam(trainable, lr=LR_START)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SCHED_T, eta_min=LR_END)
    start_epoch = 1
    curve: list[dict] = []
    best_val = float("inf")
    if last_path.is_file():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state"], strict=False)
        opt.load_state_dict(payload["optimizer"])
        sched.load_state_dict(payload["scheduler"])
        curve = list(payload.get("curve", []))
        best_val = float(payload.get("best_val", float("inf")))
        start_epoch = int(payload["epoch"]) + 1
        print(f"{family}  seed {seed}  resume epoch {start_epoch}  val {best_val:.6f}", flush=True)
    else:
        n_win = packed["p"].shape[0] if family == "quad_dynamics" else len(train_x)
        print(
            f"{family}  seed {seed}  params {n_params(model)}  windows {n_win}  "
            f"stride {STRIDE}  Adam {LR_START:g}→{LR_END:g}  WeightedMSE",
            flush=True,
        )

    if family == "quad_dynamics":
        n = packed["p"].shape[0]
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            order = torch.randperm(n)
            train_loss = 0.0
            batches = 0
            for start in range(0, n, BATCH):
                index = order[start : start + BATCH]
                pred = roll_dynamics(model, packed, index)
                loss = scaled_loss(pred, packed["y"][index], x_scale, weights)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += float(loss.detach())
                batches += 1
            sched.step()
            train_mean = train_loss / max(batches, 1)
            val_mean = _split_loss_dynamics(model, packed_val, x_scale, weights)
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
    else:
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
            val_mean = _split_loss_state(model, val_x, val_u, val_y, x_scale, weights, device)
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
    model.load_state_dict(payload["state"], strict=False)
    model.eval()
    return model


def load_trained(family: str, seed: int, device: torch.device | None = None, symmetric: bool = True) -> torch.nn.Module:
    device = train_device() if device is None else device
    path = run_dir(family, seed) / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(family, symmetric=symmetric)
    if family != "quad_dynamics":
        train, _val, _test = load_idsia_prepared()
        stats = _prepare_state(family, train, _val, device)
        model = _configure(model, family, stats, device)
    else:
        train, _val, _test = load_idsia_prepared()
        mean, scale = feat_stats(train)
        model = model.to(device)
        model.set_stats(mean.to(device), scale.to(device))
    model.load_state_dict(payload["state"], strict=False)
    model.to(device)
    model.eval()
    return model


def list_seeds(family: str) -> list[int]:
    folder = run_dir(family, 0).parent
    if not folder.is_dir():
        return []
    seeds = []
    for child in sorted(folder.glob("seed_*")):
        if (child / "best.pt").is_file():
            seeds.append(int(child.name.split("_")[1]))
    return seeds
