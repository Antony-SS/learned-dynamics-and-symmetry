"""Parity MLP, 200-step SE(3) loss, 20-step Vicon warm history then predicted state.

Fair memory: fill a length-20 delay line from Vicon (50 ms, same as GRU / NeuroBEM),
then slide in predicted (Rᵀe₃, v_body, ω) + logged Ω. No Vicon ω after the seed
except what ages out of the buffer over the first 20 steps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quadrotor.sequence import _mirror_acceleration, _mirror_features
from quadrotor.neurobem.mlp_fullstate import (
    BATCH,
    DEFAULT_ROOT,
    EPOCHS,
    EVAL_H,
    HIDDEN,
    OUT,
    RATE_HZ,
    STRIDE,
    _load,
    _se3_exp_update,
    _split,
    _state_scales,
    _windows,
    quotient_features,
    _so3_log,
)

HISTORY = 20
TRAIN_H = 200
NAME = "quotient-parity-h200-hist20"


class HistoryParityMLP(torch.nn.Module):
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

    def raw(self, stacked: torch.Tensor) -> torch.Tensor:
        return self.net(stacked)

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
        raw = self.raw(orbit).reshape(4, batch, 6)
        total = raw[0]
        for index, name in enumerate(names[1:], start=1):
            total = total + _mirror_acceleration(raw[index], name)
        return total / 4.0


def _vicon_feat(flight: dict, index: int) -> np.ndarray:
    index = int(np.clip(index, 0, len(flight["dt"]) - 1))
    return quotient_features(
        torch.as_tensor(flight["vel_body"][index : index + 1]),
        torch.as_tensor(flight["omega"][index : index + 1]),
        torch.as_tensor(flight["rotation"][index : index + 1]),
        torch.as_tensor(flight["motors"][index : index + 1]),
    )[0].numpy()


def _feature_stats(flights: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [
        quotient_features(
            torch.as_tensor(flight["vel_body"][:-1]),
            torch.as_tensor(flight["omega"][:-1]),
            torch.as_tensor(flight["rotation"][:-1]),
            torch.as_tensor(flight["motors"][:-1]),
        )
        for flight in flights
    ]
    x = torch.cat(rows, dim=0)
    return x.mean(0), x.std(0).clamp_min(1e-6)


def _warm_history(flights: list[dict], batch: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
    rows = []
    for fi, s in batch:
        slots = [_vicon_feat(flights[fi], s - HISTORY + 1 + j) for j in range(HISTORY)]
        rows.append(np.stack(slots, axis=0))
    return torch.as_tensor(np.stack(rows), device=device, dtype=torch.float32)


def _roll(
    model: HistoryParityMLP,
    *,
    position: torch.Tensor,
    rotation: torch.Tensor,
    vel_body: torch.Tensor,
    omega: torch.Tensor,
    motors: torch.Tensor,
    dt: torch.Tensor,
    hist: torch.Tensor,
    x_mean: torch.Tensor,
    x_scale: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ps, rs, vs, ws = [], [], [], []
    p, r, v, w = position, rotation, vel_body, omega
    buf = hist.clone()
    for k in range(horizon):
        stacked = ((buf - x_mean) / x_scale).reshape(buf.shape[0], -1)
        acc = model(stacked)
        v = v + acc[:, 0:3] * dt[:, k].unsqueeze(-1)
        w = w + acc[:, 3:6] * dt[:, k].unsqueeze(-1)
        p, r = _se3_exp_update(p, r, v, w, dt[:, k])
        ps.append(p)
        rs.append(r)
        vs.append(torch.bmm(r, v.unsqueeze(-1)).squeeze(-1))
        ws.append(w)
        if k + 1 < horizon:
            nxt = quotient_features(v, w, r, motors[:, k + 1])
            buf = torch.cat((buf[:, 1:], nxt.unsqueeze(1)), dim=1)
    return torch.stack(ps, 1), torch.stack(rs, 1), torch.stack(vs, 1), torch.stack(ws, 1)


def train_one(
    model: HistoryParityMLP,
    flights: list[dict],
    pairs: list[tuple[int, int]],
    *,
    device: torch.device,
    x_mean: torch.Tensor,
    x_scale: torch.Tensor,
    scales: tuple[float, float, float, float],
) -> HistoryParityMLP:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(
        f"{NAME:24s}  train {TRAIN_H} steps  hist {HISTORY} ({1000.0 * HISTORY / RATE_HZ:.0f} ms warm)  "
        f"params {sum(p.numel() for p in model.parameters())}  windows {len(pairs)}",
        flush=True,
    )
    sp, sr, sv, sw = scales
    for epoch in range(EPOCHS + 1):
        model.eval() if epoch == 0 else model.train()
        order = torch.randperm(len(pairs)) if epoch else torch.arange(len(pairs))
        parts = np.zeros(4, dtype=np.float64)
        seen = 0
        for start in range(0, len(order), BATCH):
            batch = [pairs[int(i)] for i in order[start : start + BATCH]]
            p0, r0, v0, w0, mot, dt = [], [], [], [], [], []
            p_t, r_t, v_t, w_t = [], [], [], []
            for fi, s in batch:
                f = flights[fi]
                stop = s + TRAIN_H
                p0.append(f["position"][s])
                r0.append(f["rotation"][s])
                v0.append(f["vel_body"][s])
                w0.append(f["omega"][s])
                mot.append(f["motors"][s:stop])
                dt.append(f["dt"][s:stop])
                p_t.append(f["position"][s + 1 : stop + 1])
                r_t.append(f["rotation"][s + 1 : stop + 1])
                v_t.append(f["vel_world"][s + 1 : stop + 1])
                w_t.append(f["omega"][s + 1 : stop + 1])
            pred_p, pred_r, pred_v, pred_w = _roll(
                model,
                position=torch.as_tensor(np.stack(p0), device=device),
                rotation=torch.as_tensor(np.stack(r0), device=device),
                vel_body=torch.as_tensor(np.stack(v0), device=device),
                omega=torch.as_tensor(np.stack(w0), device=device),
                motors=torch.as_tensor(np.stack(mot), device=device),
                dt=torch.as_tensor(np.stack(dt), device=device),
                hist=_warm_history(flights, batch, device),
                x_mean=x_mean,
                x_scale=x_scale,
                horizon=TRAIN_H,
            )
            true_r = torch.as_tensor(np.stack(r_t), device=device)
            rel = torch.matmul(true_r.transpose(-1, -2), pred_r)
            block = torch.stack(
                (
                    (pred_p - torch.as_tensor(np.stack(p_t), device=device)).div(sp).pow(2).mean(),
                    _so3_log(rel.reshape(-1, 3, 3)).div(sr).pow(2).mean(),
                    (pred_v - torch.as_tensor(np.stack(v_t), device=device)).div(sv).pow(2).mean(),
                    (pred_w - torch.as_tensor(np.stack(w_t), device=device)).div(sw).pow(2).mean(),
                )
            )
            loss = block.mean()
            if epoch:
                opt.zero_grad()
                loss.backward()
                opt.step()
            parts += block.detach().cpu().numpy() * len(batch)
            seen += len(batch)
        mean = parts / max(seen, 1)
        print(
            f"  {NAME:24s}  epoch {epoch:02d}  "
            f"mean {mean.mean():.5f}  p {mean[0]:.5f}  R {mean[1]:.5f}  "
            f"v {mean[2]:.5f}  w {mean[3]:.5f}",
            flush=True,
        )
    model.eval()
    return model


def _eval_pairs(flights: list[dict]) -> list[tuple[int, int]]:
    pairs = []
    for fi, flight in enumerate(flights):
        last = min(len(flight["dt"]), len(flight["omega"]) - 1) - EVAL_H
        if last < 1:
            continue
        pairs.extend((fi, s) for s in range(0, last + 1, EVAL_H))
    return pairs


def _stack_eval(flights: list[dict], batch: list[tuple[int, int]], device: torch.device):
    p0, r0, v0, w0, mot, dt = [], [], [], [], [], []
    pt, rt, vt, wt = [], [], [], []
    for fi, s in batch:
        f = flights[fi]
        stop = s + EVAL_H
        p0.append(f["position"][s])
        r0.append(f["rotation"][s])
        v0.append(f["vel_body"][s])
        w0.append(f["omega"][s])
        mot.append(f["motors"][s:stop])
        dt.append(f["dt"][s:stop])
        pt.append(f["position"][s + 1 : stop + 1])
        rt.append(f["rotation"][s + 1 : stop + 1])
        vt.append(f["vel_world"][s + 1 : stop + 1])
        wt.append(f["omega"][s + 1 : stop + 1])
    return (
        torch.as_tensor(np.stack(p0), device=device),
        torch.as_tensor(np.stack(r0), device=device),
        torch.as_tensor(np.stack(v0), device=device),
        torch.as_tensor(np.stack(w0), device=device),
        torch.as_tensor(np.stack(mot), device=device),
        torch.as_tensor(np.stack(dt), device=device),
        torch.as_tensor(np.stack(pt), device=device),
        torch.as_tensor(np.stack(rt), device=device),
        torch.as_tensor(np.stack(vt), device=device),
        torch.as_tensor(np.stack(wt), device=device),
    )


def _coast_roll(p, r, v, w, dt, horizon: int):
    ps, rs, vs, ws = [], [], [], []
    for k in range(horizon):
        p, r = _se3_exp_update(p, r, v, w, dt[:, k])
        ps.append(p)
        rs.append(r)
        vs.append(torch.bmm(r, v.unsqueeze(-1)).squeeze(-1))
        ws.append(w)
    return torch.stack(ps, 1), torch.stack(rs, 1), torch.stack(vs, 1), torch.stack(ws, 1)


@torch.no_grad()
def score_model(
    model: HistoryParityMLP | None,
    flights: list[dict],
    *,
    device: torch.device,
    x_mean: torch.Tensor | None = None,
    x_scale: torch.Tensor | None = None,
    coast: bool = False,
) -> dict[str, np.ndarray]:
    keys = ("position", "rotation", "velocity", "omega")
    pairs = _eval_pairs(flights)
    totals = {k: np.zeros(EVAL_H, dtype=np.float64) for k in keys}
    if model is not None:
        model.eval()
    print(f"  scoring {len(pairs)} windows  batch {BATCH}  h={EVAL_H}", flush=True)
    for offset in range(0, len(pairs), BATCH):
        batch = pairs[offset : offset + BATCH]
        p, r, v, w, mot, dt, pt, rt, vt, wt = _stack_eval(flights, batch, device)
        if coast:
            pred_p, pred_r, pred_v, pred_w = _coast_roll(p, r, v, w, dt, EVAL_H)
        else:
            pred_p, pred_r, pred_v, pred_w = _roll(
                model,
                position=p,
                rotation=r,
                vel_body=v,
                omega=w,
                motors=mot,
                dt=dt,
                hist=_warm_history(flights, batch, device),
                x_mean=x_mean,
                x_scale=x_scale,
                horizon=EVAL_H,
            )
        rel = torch.matmul(rt.transpose(-1, -2), pred_r)
        totals["position"] += torch.linalg.vector_norm(pred_p - pt, dim=-1).sum(0).cpu().numpy()
        totals["rotation"] += torch.linalg.vector_norm(
            _so3_log(rel.reshape(-1, 3, 3)).reshape(rel.shape[0], EVAL_H, 3), dim=-1
        ).sum(0).cpu().numpy()
        totals["velocity"] += torch.linalg.vector_norm(pred_v - vt, dim=-1).sum(0).cpu().numpy()
        totals["omega"] += torch.linalg.vector_norm(pred_w - wt, dim=-1).sum(0).cpu().numpy()
        print(f"  test {min(offset + BATCH, len(pairs))}/{len(pairs)}", flush=True)
    n = max(len(pairs), 1)
    return {k: totals[k] / n for k in keys}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    train_paths, _, test_paths = _split(DEFAULT_ROOT)
    print("loading 400 Hz  parity + 20-step Vicon warm then predicted history", flush=True)
    train = [_load(p) for p in train_paths]
    test = [_load(p) for p in test_paths]
    pairs = _windows(train, TRAIN_H, STRIDE)
    scales = _state_scales(train)
    mean, scale = _feature_stats(train)
    mean, scale = mean.to(device), scale.to(device)
    print(f"train {len(train)}  test {len(test)}  windows {len(pairs)}")
    ckpt = OUT / f"fullstate_{NAME}.pt"
    model = HistoryParityMLP()
    if ckpt.is_file():
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state"])
        mean, scale = payload["x_mean"].to(device), payload["x_scale"].to(device)
        model.to(device).eval()
        print(f"loaded {ckpt}", flush=True)
    else:
        torch.manual_seed(0)
        train_one(model, train, pairs, device=device, x_mean=mean, x_scale=scale, scales=scales)
        torch.save(
            {
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "history": HISTORY,
                "train_h": TRAIN_H,
                "x_mean": mean.cpu(),
                "x_scale": scale.cpu(),
            },
            ckpt,
        )
        print(f"saved {ckpt}", flush=True)
    print("scoring 0.5 s official-test rollouts", flush=True)
    coast = score_model(None, test, device=device, coast=True)
    pred = score_model(model, test, device=device, x_mean=mean, x_scale=scale)
    t = np.arange(1, EVAL_H + 1) / RATE_HZ
    print(f"\n{'model':28s}  {'p':>8}  {'R':>8}  {'v':>8}  {'ω':>8}")
    for label, curve in (("coast", coast), (NAME, pred)):
        print(
            f"{label:28s}  {curve['position'][199]:8.4f}  {curve['rotation'][199]:8.4f}  "
            f"{curve['velocity'][199]:8.4f}  {curve['omega'][199]:8.4f}"
        )
    np.savez(
        OUT / f"fullstate_{NAME}_curves.npz",
        t=t,
        **{f"{k}_{NAME}": pred[k] for k in pred},
        **{f"{k}_coast": coast[k] for k in coast},
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(t, coast["omega"], "k--", lw=2.2, label="coast")
    ax.plot(t, pred["omega"], color="#2a9d8f", lw=2, label=NAME)
    ax.set_xlabel("horizon [s]")
    ax.set_ylabel("ω MAE [rad/s]")
    ax.set_title("Full SE(3) parity MLP, 20-step warm history, train 200")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    fig.tight_layout()
    fig.savefig(OUT / f"fullstate_{NAME}_omega.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / f'fullstate_{NAME}_omega.png'}")


if __name__ == "__main__":
    main()
