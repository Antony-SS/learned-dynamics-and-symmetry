"""IDSIA hist-10 Z2×Z2 quotient MLP vs physics and phys+res.

Official split, 50-step (0.5 s) rollout, 20 epochs. Does not touch idsia_parity18k.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quadrotor.evaluate import benchmark_mae
from quadrotor.idsia.baselines import load_theirs
from quadrotor.idsia.data import load_idsia_prepared
from quadrotor.fit import n_params
from quadrotor.sequence import _mirror_acceleration, _mirror_features
from quadrotor.train import _apply, _features, _so3_log

HISTORY = 10
HORIZON = 50
EPOCHS = 20
HIDDEN = 64
BATCH = 256
STRIDE = 1
OUT = ROOT / "runs" / "idsia_mlp_hist10"


class HistoryParityMLP(torch.nn.Module):
    """64-64 on 10 stacked 13-D quotient frames, four-orbit Z2×Z2."""

    def __init__(self, history: int = HISTORY, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.history = int(history)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(13 * self.history, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 6),
        )

    def _mirror_stack(self, stacked: torch.Tensor, name: str) -> torch.Tensor:
        slots = stacked.reshape(stacked.shape[0], self.history, 13)
        return torch.stack(
            [_mirror_features(slots[:, k], name) for k in range(self.history)],
            dim=1,
        ).reshape(stacked.shape[0], -1)

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        batch = stacked.shape[0]
        names = (None, "lr", "fb", "both")
        orbit = torch.cat(
            [
                stacked if name is None else self._mirror_stack(stacked, name)
                for name in names
            ],
            dim=0,
        )
        raw = self.net(orbit).reshape(4, batch, 6)
        total = raw[0]
        for index, name in enumerate(names[1:], start=1):
            total = total + _mirror_acceleration(raw[index], name)
        return total / 4.0


def _feat(flight, index: int) -> np.ndarray:
    index = int(np.clip(index, 0, flight.n_steps - 1))
    pwm_i = min(index, len(flight.pwm) - 1)
    g = flight.rotation[index].T @ np.array((0.0, 0.0, 1.0), np.float32)
    v = flight.rotation[index].T @ flight.velocity_world[index]
    return np.concatenate(
        (g, v, flight.omega_body[index], flight.pwm[pwm_i])
    ).astype(np.float32)


def _feat_t(
    rotation: torch.Tensor,
    vel_world: torch.Tensor,
    omega: torch.Tensor,
    pwm: torch.Tensor,
) -> torch.Tensor:
    dummy = type("M", (), {"reads_pose": False, "reads_voltage": False, "reads_height": False})()
    return _features(dummy, None, rotation, vel_world, omega, pwm, None)


def _starts(flights, horizon: int, stride: int) -> list[tuple[int, int]]:
    return [
        (i, s)
        for i, f in enumerate(flights)
        for s in range(0, f.n_steps - horizon, stride)
    ]


def _scales(flights) -> tuple[float, float, float, float]:
    p = np.concatenate([f.position for f in flights])
    v = np.concatenate([f.velocity_world for f in flights])
    w = np.concatenate([f.omega_body for f in flights])
    r = _so3_log(torch.as_tensor(np.concatenate([f.rotation for f in flights]))).numpy()
    return (
        max(float(np.std(p)), 1e-6),
        max(float(np.std(r)), 1e-6),
        max(float(np.std(v)), 1e-6),
        max(float(np.std(w)), 1e-6),
    )


def _feat_stats(flights) -> tuple[torch.Tensor, torch.Tensor]:
    rows = np.stack([_feat(f, k) for f in flights for k in range(f.n_steps)])
    return (
        torch.as_tensor(rows.mean(0), dtype=torch.float32),
        torch.as_tensor(np.clip(rows.std(0), 1e-6, None), dtype=torch.float32),
    )


def _warm(flights, batch, device) -> torch.Tensor:
    rows = []
    for fi, s in batch:
        rows.append(np.stack([_feat(flights[fi], s - HISTORY + 1 + j) for j in range(HISTORY)]))
    return torch.as_tensor(np.stack(rows), device=device, dtype=torch.float32)


def _roll(model, p, r, v_w, w, pwm, dt, hist, x_mean, x_scale, horizon):
    ps, rs, vs, ws = [], [], [], []
    buf = hist.clone()
    for k in range(horizon):
        acc = model(((buf - x_mean) / x_scale).reshape(buf.shape[0], -1))
        p, r, v_w, w = _apply(p, r, v_w, w, acc, dt[:, k])
        ps.append(p)
        rs.append(r)
        vs.append(v_w)
        ws.append(w)
        if k + 1 < horizon:
            nxt = _feat_t(r, v_w, w, pwm[:, k + 1]).unsqueeze(1)
            buf = torch.cat((buf[:, 1:], nxt), dim=1)
    return torch.stack(ps, 1), torch.stack(rs, 1), torch.stack(vs, 1), torch.stack(ws, 1)


def train_mlp(train, pairs, device, x_mean, x_scale, scales):
    model = HistoryParityMLP().to(device)
    print(f"hist-10 parity MLP  params {n_params(model)}  windows {len(pairs)}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sp, sr, sv, sw = scales
    for epoch in range(EPOCHS + 1):
        model.eval() if epoch == 0 else model.train()
        order = torch.randperm(len(pairs)) if epoch else torch.arange(len(pairs))
        parts = np.zeros(4)
        seen = 0
        for start in range(0, len(order), BATCH):
            batch = [pairs[int(i)] for i in order[start : start + BATCH]]
            p0, r0, v0, w0, pwm, dt = [], [], [], [], [], []
            pt, rt, vt, wt = [], [], [], []
            for fi, s in batch:
                f = train[fi]
                stop = s + HORIZON
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
            pred_p, pred_r, pred_v, pred_w = _roll(
                model,
                torch.as_tensor(np.stack(p0), device=device),
                torch.as_tensor(np.stack(r0), device=device),
                torch.as_tensor(np.stack(v0), device=device),
                torch.as_tensor(np.stack(w0), device=device),
                torch.as_tensor(np.stack(pwm), device=device),
                torch.as_tensor(np.stack(dt), device=device),
                _warm(train, batch, device),
                x_mean,
                x_scale,
                HORIZON,
            )
            rel = torch.matmul(
                torch.as_tensor(np.stack(rt), device=device).transpose(-1, -2), pred_r
            )
            block = torch.stack(
                (
                    (pred_p - torch.as_tensor(np.stack(pt), device=device)).div(sp).pow(2).mean(),
                    _so3_log(rel.reshape(-1, 3, 3)).div(sr).pow(2).mean(),
                    (pred_v - torch.as_tensor(np.stack(vt), device=device)).div(sv).pow(2).mean(),
                    (pred_w - torch.as_tensor(np.stack(wt), device=device)).div(sw).pow(2).mean(),
                )
            )
            if epoch:
                opt.zero_grad()
                block.mean().backward()
                opt.step()
            parts += block.detach().cpu().numpy() * len(batch)
            seen += len(batch)
        m = parts / max(seen, 1)
        print(
            f"  hist-10  epoch {epoch:02d}  mean {m.mean():.5f}  "
            f"p {m[0]:.5f}  R {m[1]:.5f}  v {m[2]:.5f}  w {m[3]:.5f}",
            flush=True,
        )
    model.eval()
    return model


@torch.no_grad()
def score_mlp(model, flights, device, x_mean, x_scale) -> dict[str, np.ndarray]:
    keys = ("position", "velocity", "rotation", "omega")
    pairs = [
        (fi, s)
        for fi, flight in enumerate(flights)
        for s in range(0, flight.n_steps - HORIZON + 1)
    ]
    totals = {k: np.zeros(HORIZON) for k in keys}
    model.eval()
    print(f"  scoring Melon  {len(pairs)} windows  batch {BATCH}", flush=True)
    for offset in range(0, len(pairs), BATCH):
        batch = pairs[offset : offset + BATCH]
        p0, r0, v0, w0, pwm, dt = [], [], [], [], [], []
        pt, rt, vt, wt = [], [], [], []
        for fi, s in batch:
            f = flights[fi]
            stop = s + HORIZON
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
        pred_p, pred_r, pred_v, pred_w = _roll(
            model,
            torch.as_tensor(np.stack(p0), device=device),
            torch.as_tensor(np.stack(r0), device=device),
            torch.as_tensor(np.stack(v0), device=device),
            torch.as_tensor(np.stack(w0), device=device),
            torch.as_tensor(np.stack(pwm), device=device),
            torch.as_tensor(np.stack(dt), device=device),
            _warm(flights, batch, device),
            x_mean,
            x_scale,
            HORIZON,
        )
        rel = torch.matmul(torch.as_tensor(np.stack(rt), device=device).transpose(-1, -2), pred_r)
        totals["position"] += torch.linalg.vector_norm(
            pred_p - torch.as_tensor(np.stack(pt), device=device), dim=-1
        ).sum(0).cpu().numpy()
        totals["rotation"] += torch.linalg.vector_norm(
            _so3_log(rel.reshape(-1, 3, 3)).reshape(rel.shape[0], HORIZON, 3), dim=-1
        ).sum(0).cpu().numpy()
        totals["velocity"] += torch.linalg.vector_norm(
            pred_v - torch.as_tensor(np.stack(vt), device=device), dim=-1
        ).sum(0).cpu().numpy()
        totals["omega"] += torch.linalg.vector_norm(
            pred_w - torch.as_tensor(np.stack(wt), device=device), dim=-1
        ).sum(0).cpu().numpy()
        print(f"  melon {min(offset + BATCH, len(pairs))}/{len(pairs)}", flush=True)
    return {k: totals[k] / max(len(pairs), 1) for k in keys}


def _row(name: str, mae: dict[str, np.ndarray]) -> None:
    print(
        f"{name:22s}  p {mae['position'][9]:.4f}/{mae['position'][49]:.4f}  "
        f"R {mae['rotation'][9]:.4f}/{mae['rotation'][49]:.4f}  "
        f"v {mae['velocity'][9]:.4f}/{mae['velocity'][49]:.4f}  "
        f"w {mae['omega'][9]:.4f}/{mae['omega'][49]:.4f}"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    train, val, test = load_idsia_prepared()
    pairs = _starts(train, HORIZON, STRIDE)
    print(f"IDSIA  train {len(train)}  val {len(val)}  melon {len(test)}  windows {len(pairs)}")
    mean, scale = _feat_stats(train)
    mean, scale = mean.to(device), scale.to(device)
    ckpt = OUT / "hist10_parity.pt"
    if ckpt.is_file():
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = HistoryParityMLP()
        model.load_state_dict(payload["state"])
        model.to(device).eval()
        mean = payload["x_mean"].to(device)
        scale = payload["x_scale"].to(device)
        print(f"loaded {ckpt}  params {n_params(model)}")
    else:
        torch.manual_seed(0)
        model = train_mlp(train, pairs, device, mean, scale, _scales(train))
        torch.save(
            {
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "x_mean": mean.cpu(),
                "x_scale": scale.cpu(),
                "history": HISTORY,
                "params": n_params(model),
            },
            ckpt,
        )
        print(f"saved {ckpt}")

    print("\nMelon MAE  h10 / h50  (0.10 s / 0.50 s)  stride 1")
    print(f"{'model':22s}  p h10/h50          R h10/h50          v h10/h50          w h10/h50")
    mlp_mae = score_mlp(model, test, device, mean, scale)
    _row(f"hist-10 MLP ({n_params(model)})", mlp_mae)

    theirs = load_theirs(train_flights=train, device=device)
    for name in ("physics", "phys+res"):
        if name in theirs:
            mae = benchmark_mae(theirs[name], test, horizon=HORIZON, start_stride=1)
            _row(name, mae)



if __name__ == "__main__":
    main()
