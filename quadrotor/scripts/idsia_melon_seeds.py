"""5-seed Melon MAE for the hist-10 table.

Retrains phys+res, hist-10 Z2×Z2, hist-10 quotient, and hist-10 world.
Physics is analytic (one score). Does not touch idsia_parity18k or the
published IDSIA phys+res checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quadrotor.evaluate import benchmark_mae
from quadrotor.fit import n_params, train_device
from quadrotor.geometry import so3_log_batch
from quadrotor.idsia.baselines import PhysQuadModel, PhysResQuadModel, ResidualQuadModel, train_phys_res
from quadrotor.idsia.data import load_idsia_prepared
from quadrotor.idsia.history_mlp import (
    HORIZON,
    TRAIN_STRIDE,
    HistoryQuotientMLP,
    save_model,
    score_history,
    starts,
    state_scales,
    train_history,
)
from quadrotor.train import _apply, _features, _so3_log

OUT = ROOT / "runs" / "idsia_melon_seeds_s5"
PRIOR_Z2 = ROOT / "runs" / "idsia_mlp_hist10_stride5" / "hist10_parity_stride5.pt"
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 20
HS = (("h1", 0), ("h10", 9), ("h50", 49))
KEYS = ("position", "rotation", "velocity", "omega")


class HistoryWorldMLP(torch.nn.Module):
    """64-64 on stacked 16-D world frames (p, v, so3_log R, ω, Ω)."""

    reads_pose = True
    reads_voltage = False
    reads_height = False

    def __init__(self, history: int = 10, hidden: int = 64) -> None:
        super().__init__()
        self.history = int(history)
        self.dim = 16
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.dim * self.history, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 6),
        )
        self.register_buffer("x_mean", torch.zeros(self.dim))
        self.register_buffer("x_scale", torch.ones(self.dim))

    def set_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        with torch.no_grad():
            self.x_mean.copy_(mean.reshape(self.dim).to(self.x_mean))
            self.x_scale.copy_(scale.reshape(self.dim).to(self.x_scale).clamp_min(1e-6))

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        return self.net(stacked)


def world_row(flight, index: int) -> np.ndarray:
    index = int(np.clip(index, 0, flight.n_steps - 1))
    pwm_i = min(index, len(flight.pwm) - 1)
    so3 = so3_log_batch(flight.rotation[index : index + 1])[0]
    return np.concatenate(
        (
            flight.position[index],
            flight.velocity_world[index],
            so3.astype(np.float32),
            flight.omega_body[index],
            flight.pwm[pwm_i],
        )
    ).astype(np.float32)


def world_feat_t(position, rotation, vel_world, omega, pwm) -> torch.Tensor:
    dummy = type(
        "M",
        (),
        {"reads_pose": True, "reads_voltage": False, "reads_height": False},
    )()
    return _features(dummy, position, rotation, vel_world, omega, pwm, None)


def world_features(flight) -> np.ndarray:
    n = flight.n_steps
    so3 = so3_log_batch(flight.rotation[:n]).astype(np.float32)
    return np.concatenate(
        (
            flight.position[:n],
            flight.velocity_world[:n],
            so3,
            flight.omega_body[:n],
            flight.pwm,
        ),
        axis=1,
    ).astype(np.float32)


def world_stats(flights):
    rows = np.concatenate([world_features(f) for f in flights], axis=0)
    return (
        torch.as_tensor(rows.mean(0), dtype=torch.float32),
        torch.as_tensor(np.clip(rows.std(0), 1e-6, None), dtype=torch.float32),
    )


def pack_world(flights, pairs, history: int, device: torch.device) -> dict[str, torch.Tensor]:
    feats = [torch.as_tensor(world_features(f), device=device) for f in flights]
    pos = [torch.as_tensor(f.position, device=device) for f in flights]
    rot = [torch.as_tensor(f.rotation, device=device) for f in flights]
    vel = [torch.as_tensor(f.velocity_world, device=device) for f in flights]
    ome = [torch.as_tensor(f.omega_body, device=device) for f in flights]
    pwm = [torch.as_tensor(f.pwm, device=device) for f in flights]
    dt = [torch.as_tensor(f.dt, device=device) for f in flights]
    p0, r0, v0, w0, u, dts, pt, rt, vt, wt, hist = [], [], [], [], [], [], [], [], [], [], []
    for fi, s in pairs:
        n = feats[fi].shape[0]
        idx = torch.arange(s - history + 1, s + 1, device=device).clamp(0, n - 1)
        hist.append(feats[fi][idx])
        stop = s + HORIZON
        p0.append(pos[fi][s])
        r0.append(rot[fi][s])
        v0.append(vel[fi][s])
        w0.append(ome[fi][s])
        u.append(pwm[fi][s:stop])
        dts.append(dt[fi][s:stop])
        pt.append(pos[fi][s + 1 : stop + 1])
        rt.append(rot[fi][s + 1 : stop + 1])
        vt.append(vel[fi][s + 1 : stop + 1])
        wt.append(ome[fi][s + 1 : stop + 1])
    return {
        "p": torch.stack(p0),
        "r": torch.stack(r0),
        "v": torch.stack(v0),
        "w": torch.stack(w0),
        "pwm": torch.stack(u),
        "dt": torch.stack(dts),
        "pt": torch.stack(pt),
        "rt": torch.stack(rt),
        "vt": torch.stack(vt),
        "wt": torch.stack(wt),
        "hist": torch.stack(hist),
    }


def world_roll(model, p, r, v_w, w, pwm, dt, hist, horizon: int = HORIZON):
    ps, rs, vs, ws = [], [], [], []
    buf = hist.clone()
    for k in range(horizon):
        flat = ((buf - model.x_mean) / model.x_scale).reshape(buf.shape[0], -1)
        acc = model(flat)
        p, r, v_w, w = _apply(p, r, v_w, w, acc, dt[:, k])
        ps.append(p)
        rs.append(r)
        vs.append(v_w)
        ws.append(w)
        if k + 1 < horizon:
            nxt = world_feat_t(p, r, v_w, w, pwm[:, k + 1]).unsqueeze(1)
            buf = torch.cat((buf[:, 1:], nxt), dim=1)
    return torch.stack(ps, 1), torch.stack(rs, 1), torch.stack(vs, 1), torch.stack(ws, 1)


def _world_block(model, packed, scales, sl):
    pred_p, pred_r, pred_v, pred_w = world_roll(
        model,
        packed["p"][sl],
        packed["r"][sl],
        packed["v"][sl],
        packed["w"][sl],
        packed["pwm"][sl],
        packed["dt"][sl],
        packed["hist"][sl],
    )
    rel = torch.matmul(packed["rt"][sl].transpose(-1, -2), pred_r)
    sp, sr, sv, sw = scales
    return torch.stack(
        (
            (pred_p - packed["pt"][sl]).div(sp).pow(2).mean(),
            _so3_log(rel.reshape(-1, 3, 3)).div(sr).pow(2).mean(),
            (pred_v - packed["vt"][sl]).div(sv).pow(2).mean(),
            (pred_w - packed["wt"][sl]).div(sw).pow(2).mean(),
        )
    )


@torch.no_grad()
def _world_split_loss(model, packed, scales) -> np.ndarray:
    model.eval()
    parts = np.zeros(4)
    seen = 0
    n = packed["p"].shape[0]
    for start in range(0, n, 256):
        sl = slice(start, start + 256)
        block = _world_block(model, packed, scales, sl)
        count = min(256, n - start)
        parts += block.detach().cpu().numpy() * count
        seen += count
    return parts / max(seen, 1)


def train_world(train, val, device, seed: int) -> HistoryWorldMLP:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean, scale = world_stats(train)
    model = HistoryWorldMLP().to(device)
    model.set_stats(mean.to(device), scale.to(device))
    pairs = starts(train, stride=TRAIN_STRIDE)
    packed = pack_world(train, pairs, 10, device)
    packed_val = pack_world(val, starts(val, stride=TRAIN_STRIDE), 10, device)
    print(
        f"hist-10 world  params {n_params(model)}  windows {len(pairs)}  "
        f"train stride {TRAIN_STRIDE}  seed {seed}",
        flush=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scales = state_scales(train)
    n = len(pairs)
    curve = []
    for epoch in range(1, EPOCHS + 1):
        print(f"  world  seed {seed}  epoch {epoch:02d}  starting  {n} windows", flush=True)
        model.train()
        order = torch.randperm(n)
        parts = np.zeros(4)
        seen = 0
        for start in range(0, n, 256):
            index = order[start : start + 256]
            block = _world_block(model, packed, scales, index)
            opt.zero_grad()
            block.mean().backward()
            opt.step()
            parts += block.detach().cpu().numpy() * int(index.numel())
            seen += int(index.numel())
        m = parts / max(seen, 1)
        vm = _world_split_loss(model, packed_val, scales)
        curve.append(
            {
                "epoch": epoch,
                "train": float(m.mean()),
                "val": float(vm.mean()),
                "train_p": float(m[0]),
                "val_p": float(vm[0]),
            }
        )
        print(
            f"  world  seed {seed}  epoch {epoch:02d}  train {m.mean():.5f}  "
            f"val {vm.mean():.5f}  p {m[0]:.5f}  R {m[1]:.5f}  v {m[2]:.5f}  w {m[3]:.5f}",
            flush=True,
        )
    model.train_curve = curve
    model.eval()
    return model


@torch.no_grad()
def score_world(model, flights, device) -> dict[str, np.ndarray]:
    pairs = [
        (fi, s)
        for fi, flight in enumerate(flights)
        for s in range(0, flight.n_steps - HORIZON + 1)
    ]
    packed = pack_world(flights, pairs, model.history, device)
    totals = {k: np.zeros(HORIZON) for k in KEYS}
    print(f"  scoring world Melon  {len(pairs)} windows", flush=True)
    for offset in range(0, len(pairs), 256):
        index = slice(offset, offset + 256)
        pred_p, pred_r, pred_v, pred_w = world_roll(
            model,
            packed["p"][index],
            packed["r"][index],
            packed["v"][index],
            packed["w"][index],
            packed["pwm"][index],
            packed["dt"][index],
            packed["hist"][index],
        )
        rel = torch.matmul(packed["rt"][index].transpose(-1, -2), pred_r)
        totals["position"] += torch.linalg.vector_norm(
            pred_p - packed["pt"][index], dim=-1
        ).sum(0).cpu().numpy()
        totals["rotation"] += (
            torch.linalg.vector_norm(
                _so3_log(rel.reshape(-1, 3, 3)).reshape(rel.shape[0], -1, 3), dim=-1
            )
            .sum(0)
            .cpu()
            .numpy()
        )
        totals["velocity"] += torch.linalg.vector_norm(
            pred_v - packed["vt"][index], dim=-1
        ).sum(0).cpu().numpy()
        totals["omega"] += torch.linalg.vector_norm(
            pred_w - packed["wt"][index], dim=-1
        ).sum(0).cpu().numpy()
    return {k: v / max(len(pairs), 1) for k, v in totals.items()}


def pack(mae: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {k: [float(mae[k][i]) for _, i in HS] for k in KEYS}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_results() -> dict:
    path = OUT / "results.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def save_results(results: dict) -> None:
    write_json(OUT / "results.json", results)


def fmt_cell(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return f"{arr[0]:.4f}"
    return f"{arr.mean():.4f}±{arr.std(ddof=1):.4f}"


def print_table(results: dict) -> None:
    order = (
        "physics",
        "phys+res",
        "hist-10 Z2xZ2",
        "hist-10 quotient",
        "hist-10 world",
    )
    print("\nMelon MAE  h=1 / h=10 / h=50   mean±std over finished seeds", flush=True)
    print(f"{'model':22s}  {'p [m]':28s}  {'R [rad]':28s}  {'v [m/s]':28s}  {'ω [rad/s]'}", flush=True)
    for name in order:
        if name not in results:
            continue
        seeds = results[name]
        cols = []
        for key in KEYS:
            triples = [seeds[s][key] for s in sorted(seeds)]
            cols.append(" / ".join(fmt_cell([row[i] for row in triples]) for i in range(3)))
        print(f"{name:22s}  {cols[0]:28s}  {cols[1]:28s}  {cols[2]:28s}  {cols[3]}", flush=True)
    lines = [
        "Melon MAE  h=1 / h=10 / h=50   mean±std  train stride 5",
        f"{'model':22s}  {'p [m]':28s}  {'R [rad]':28s}  {'v [m/s]':28s}  {'ω [rad/s]'}",
    ]
    for name in order:
        if name not in results:
            continue
        seeds = results[name]
        cols = []
        for key in KEYS:
            triples = [seeds[s][key] for s in sorted(seeds)]
            cols.append(" / ".join(fmt_cell([row[i] for row in triples]) for i in range(3)))
        lines.append(f"{name:22s}  {cols[0]:28s}  {cols[1]:28s}  {cols[2]:28s}  {cols[3]}")
    (OUT / "table.txt").write_text("\n".join(lines) + "\n")


def load_curves() -> dict:
    path = OUT / "curves.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def save_curves(curves: dict) -> None:
    write_json(OUT / "curves.json", curves)


def plot_curves(curves: dict) -> None:
    names = ("phys+res", "hist-10 Z2xZ2", "hist-10 quotient", "hist-10 world")
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), sharex=True)
    for ax, name in zip(axes.ravel(), names):
        seeds = curves.get(name, {})
        trains, vals = [], []
        for seed, rows in sorted(seeds.items()):
            ep = [r["epoch"] for r in rows]
            tr = [r["train"] for r in rows]
            va = [r.get("val") for r in rows]
            ax.plot(ep, tr, color="#4c78a8", alpha=0.25, lw=1)
            if any(v is not None for v in va):
                ax.plot(ep, va, color="#f58518", alpha=0.25, lw=1)
            trains.append(tr)
            if all(v is not None for v in va):
                vals.append(va)
        if trains:
            ax.plot(range(1, len(trains[0]) + 1), np.mean(trains, 0), color="#4c78a8", lw=2.2, label="train")
        if vals:
            ax.plot(range(1, len(vals[0]) + 1), np.mean(vals, 0), color="#f58518", lw=2.2, label="val")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("rollout MSE")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("epoch")
    axes[1, 1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(OUT / "train_val_curves.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT / 'train_val_curves.png'}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = train_device()
    train, val, test = load_idsia_prepared()
    results = load_results()
    curves = load_curves()
    print(
        f"IDSIA Melon 5-seed  train stride {TRAIN_STRIDE}  "
        f"train {len(train)}  val {len(val)}  test {len(test)}  "
        f"device {device}  -> {OUT}",
        flush=True,
    )

    if "physics" not in results:
        phys = PhysQuadModel(zero_omega_dot=True).to(device).eval()
        mae = benchmark_mae(phys, test, horizon=HORIZON, start_stride=1)
        results["physics"] = {"analytic": pack(mae)}
        save_results(results)
        print("scored physics", flush=True)

    for seed in SEEDS:
        key = str(seed)
        folder = OUT / "phys_res"
        ckpt = folder / f"seed{seed}.pt"
        if "phys+res" in results and key in results["phys+res"] and key in curves.get("phys+res", {}):
            print(f"phys+res  seed {seed}  already scored", flush=True)
            continue
        if ckpt.is_file():
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            phys = PhysQuadModel(zero_omega_dot=True)
            residual = ResidualQuadModel(hidden_dim=64, num_layers=5)
            model = PhysResQuadModel(phys, residual)
            model.load_state_dict(payload["state"])
            model.to(device).eval()
            print(f"loaded {ckpt}", flush=True)
        else:
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"training phys+res  seed {seed}", flush=True)
            model = train_phys_res(
                train, val, epochs=EPOCHS, stride=TRAIN_STRIDE, device=device
            )
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "seed": seed,
                    "curve": list(getattr(model, "train_curve", [])),
                },
                ckpt,
            )
            curves.setdefault("phys+res", {})[key] = list(getattr(model, "train_curve", []))
            save_curves(curves)
        mae = benchmark_mae(model, test, horizon=HORIZON, start_stride=1)
        results.setdefault("phys+res", {})[key] = pack(mae)
        save_results(results)
        print(f"scored phys+res  seed {seed}", flush=True)

    for name, symmetric, tag in (
        ("hist-10 Z2xZ2", True, "hist-10 Z2×Z2"),
        ("hist-10 quotient", False, "hist-10 quotient"),
    ):
        for seed in SEEDS:
            key = str(seed)
            folder = OUT / name.replace(" ", "_")
            ckpt = folder / f"seed{seed}.pt"
            if name in results and key in results[name] and key in curves.get(name, {}):
                print(f"{name}  seed {seed}  already scored", flush=True)
                continue
            if ckpt.is_file() and key in curves.get(name, {}):
                payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                model = HistoryQuotientMLP(history=10, symmetric=symmetric)
                model.load_state_dict(payload["state"], strict=False)
                model.set_stats(payload["x_mean"], payload["x_scale"])
                model.to(device).eval()
                print(f"loaded {ckpt}", flush=True)
            else:
                print(f"training {name}  seed {seed}", flush=True)
                model = train_history(
                    train,
                    history=10,
                    symmetric=symmetric,
                    device=device,
                    seed=seed,
                    tag=tag,
                    stride=TRAIN_STRIDE,
                    val=val,
                )
                save_model(model, ckpt)
                curves.setdefault(name, {})[key] = list(getattr(model, "train_curve", []))
                save_curves(curves)
            mae = score_history(model, test, device)
            results.setdefault(name, {})[key] = pack(mae)
            save_results(results)
            print(f"scored {name}  seed {seed}", flush=True)

    for seed in SEEDS:
        key = str(seed)
        name = "hist-10 world"
        folder = OUT / "hist-10_world"
        ckpt = folder / f"seed{seed}.pt"
        if name in results and key in results[name] and key in curves.get(name, {}):
            print(f"{name}  seed {seed}  already scored", flush=True)
            continue
        if ckpt.is_file() and key in curves.get(name, {}):
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            model = HistoryWorldMLP()
            model.load_state_dict(payload["state"])
            model.set_stats(payload["x_mean"], payload["x_scale"])
            model.to(device).eval()
            print(f"loaded {ckpt}", flush=True)
        else:
            print(f"training {name}  seed {seed}", flush=True)
            model = train_world(train, val, device, seed)
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "x_mean": model.x_mean.detach().cpu(),
                    "x_scale": model.x_scale.detach().cpu(),
                    "seed": seed,
                    "curve": list(getattr(model, "train_curve", [])),
                },
                ckpt,
            )
            curves.setdefault(name, {})[key] = list(getattr(model, "train_curve", []))
            save_curves(curves)
        mae = score_world(model, test, device)
        results.setdefault(name, {})[key] = pack(mae)
        save_results(results)
        print(f"scored {name}  seed {seed}", flush=True)

    print_table(results)
    plot_curves(curves)
    print(f"wrote {OUT / 'results.json'}  {OUT / 'curves.json'}", flush=True)


if __name__ == "__main__":
    main()
