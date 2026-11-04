"""Rao et al. decoupled protocol, our ~18k leftover (not their 5.2M nets).

100 Hz NeuroBEM, warm 20 (0.2 s), train unroll 10, eval 60 (0.6 s).
Velocity net: teacher-force logged R every step; predict (a, α); integrate v, ω.
Attitude net: teacher-force logged v, ω; predict α; integrate R with GT ω + α.

delta_z = mean_t(||v-vhat||^2 + ||w-what||^2)
delta_q = mean_t geodesic attitude [rad]

Does not touch out/parity18k.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quadrotor.fit import SymmetricModel, n_params, train_device
from quadrotor.neurobem.data import load_neurobem_prepared
from quadrotor.train import (
    _configure_input_normalization,
    _features,
    _so3_exp,
    _so3_log,
    module_device,
    prime_hidden,
)

OUT = Path(__file__).resolve().parent / "out" / "nyu_decoupled18k"
EVERY = 4
HISTORY = 20
TRAIN_H = 10
EVAL_H = 60
TRAIN_STRIDE = 20
EVAL_STRIDE = 10
EPOCHS = 8
BATCH = 256
LR = 1e-3


def at_100hz(flight):
    n = (flight.n_steps // EVERY) * EVERY
    if n < EVERY * (HISTORY + EVAL_H):
        return replace(flight, pwm=flight.pwm[:0], dt=flight.dt[:0])
    sl = slice(0, n, EVERY)
    return replace(
        flight,
        position=flight.position[: n + 1 : EVERY],
        rotation=flight.rotation[: n + 1 : EVERY],
        velocity_world=flight.velocity_world[: n + 1 : EVERY],
        omega_body=flight.omega_body[: n + 1 : EVERY],
        pwm=flight.pwm[sl],
        voltage=flight.voltage[sl],
        dt=flight.dt[:n].reshape(-1, EVERY).sum(axis=1).astype(np.float32),
    )


def keep(flights, horizon: int):
    return [f for f in (at_100hz(f) for f in flights) if f.n_steps > HISTORY + horizon]


def windows(flights, horizon: int, stride: int) -> list[tuple[int, int]]:
    pairs = []
    for i, flight in enumerate(flights):
        last = flight.n_steps - horizon
        for start in range(HISTORY, last + 1, stride):
            pairs.append((i, start))
    return pairs


def tensors(flights, batch, horizon: int, device: torch.device):
    pwm, dt, p, r, v, w = [], [], [], [], [], []
    for fi, s in batch:
        f = flights[fi]
        stop = s + horizon
        pwm.append(f.pwm[s:stop])
        dt.append(f.dt[s:stop])
        p.append(f.position[s : stop + 1])
        r.append(f.rotation[s : stop + 1])
        v.append(f.velocity_world[s : stop + 1])
        w.append(f.omega_body[s : stop + 1])
    return (
        torch.as_tensor(np.stack(pwm), device=device),
        torch.as_tensor(np.stack(dt), device=device),
        torch.as_tensor(np.stack(p), device=device),
        torch.as_tensor(np.stack(r), device=device),
        torch.as_tensor(np.stack(v), device=device),
        torch.as_tensor(np.stack(w), device=device),
    )


def roll_velocity(model, pwm, dt, r_gt, v, w, hidden):
    vs, ws = [], []
    dummy = v.new_zeros(v.shape[0], 3)
    for t in range(pwm.shape[1]):
        rot = r_gt[:, t]
        acc, hidden = model.step(_features(model, dummy, rot, v, w, pwm[:, t]), hidden)
        step = dt[:, t].unsqueeze(-1)
        v = v + torch.bmm(rot, acc[:, 0:3].unsqueeze(-1)).squeeze(-1) * step
        w = w + acc[:, 3:6] * step
        vs.append(v)
        ws.append(w)
    return torch.stack(vs, 1), torch.stack(ws, 1)


def roll_attitude(model, pwm, dt, r, v_gt, w_gt, hidden):
    rs = []
    dummy = r.new_zeros(r.shape[0], 3)
    for t in range(pwm.shape[1]):
        acc, hidden = model.step(
            _features(model, dummy, r, v_gt[:, t], w_gt[:, t], pwm[:, t]), hidden
        )
        step = dt[:, t].unsqueeze(-1)
        phi = (w_gt[:, t] + 0.5 * acc[:, 3:6] * step) * step
        r = torch.bmm(r, _so3_exp(phi))
        rs.append(r)
    return torch.stack(rs, 1)


@torch.no_grad()
def score(vel, att, flights, horizon: int = EVAL_H, stride: int = EVAL_STRIDE) -> dict:
    pairs = windows(flights, horizon, stride)
    device = module_device(vel)
    dz = 0.0
    dq = 0.0
    seen = 0
    vel.eval()
    att.eval()
    for offset in range(0, len(pairs), BATCH):
        batch = pairs[offset : offset + BATCH]
        pwm, dt, _p, r, v, w = tensors(flights, batch, horizon, device)
        v_hat, w_hat = roll_velocity(
            vel, pwm, dt, r, v[:, 0], w[:, 0], prime_hidden(vel, flights, batch, HISTORY)
        )
        r_hat = roll_attitude(
            att, pwm, dt, r[:, 0], v, w, prime_hidden(att, flights, batch, HISTORY)
        )
        vel_sq = (v_hat - v[:, 1:]).square().sum(-1) + (w_hat - w[:, 1:]).square().sum(-1)
        rel = torch.matmul(r[:, 1:].transpose(-1, -2), r_hat)
        att_err = torch.linalg.vector_norm(_so3_log(rel.reshape(-1, 3, 3)), dim=-1)
        dz += float(vel_sq.sum().cpu())
        dq += float(att_err.sum().cpu())
        seen += vel_sq.numel()
    return {"delta_z": dz / max(seen, 1), "delta_q": dq / max(seen, 1), "windows": len(pairs)}


def train_head(name, model, flights, pairs, kind: str) -> None:
    device = module_device(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(pairs)).tolist()
        model.train()
        total = 0.0
        seen = 0
        for offset in range(0, len(order), BATCH):
            batch = [pairs[i] for i in order[offset : offset + BATCH]]
            pwm, dt, _p, r, v, w = tensors(flights, batch, TRAIN_H, device)
            hidden = prime_hidden(model, flights, batch, HISTORY)
            if kind == "vel":
                v_hat, w_hat = roll_velocity(model, pwm, dt, r, v[:, 0], w[:, 0], hidden)
                loss = (v_hat - v[:, 1:]).square().mean() + (w_hat - w[:, 1:]).square().mean()
            else:
                r_hat = roll_attitude(model, pwm, dt, r[:, 0], v, w, hidden)
                rel = torch.matmul(r[:, 1:].transpose(-1, -2), r_hat)
                loss = _so3_log(rel.reshape(-1, 3, 3)).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            total += float(loss.detach()) * len(batch)
            seen += len(batch)
        metric = total / max(seen, 1)
        if metric < best:
            best = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"{name:4s}  epoch {epoch:02d}/{EPOCHS}  train {metric:.5f}  best {best:.5f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    model.to(device)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = train_device()
    raw_train, raw_val, raw_test = load_neurobem_prepared()
    train, val, test = keep(raw_train, EVAL_H), keep(raw_val, EVAL_H), keep(raw_test, EVAL_H)
    pairs = windows(train, TRAIN_H, TRAIN_STRIDE)
    print(
        f"NYU decoupled  100 Hz  warm {HISTORY}  train U={TRAIN_H}  eval T={EVAL_H}  "
        f"train flights {len(train)}  val {len(val)}  test {len(test)}  "
        f"windows {len(pairs)}  device {device}",
        flush=True,
    )

    vel = SymmetricModel(include_voltage=False).to(device)
    att = SymmetricModel(include_voltage=False).to(device)
    _configure_input_normalization(vel, train)
    att.load_state_dict(vel.state_dict())
    print(f"vel {n_params(vel):,}   att {n_params(att):,}   total {n_params(vel)+n_params(att):,}", flush=True)

    train_head("vel", vel, train, pairs, "vel")
    train_head("att", att, train, pairs, "att")
    torch.save(
        {
            "vel": {k: v.detach().cpu() for k, v in vel.state_dict().items()},
            "att": {k: v.detach().cpu() for k, v in att.state_dict().items()},
            "history": HISTORY,
            "train_h": TRAIN_H,
            "eval_h": EVAL_H,
            "params_each": n_params(vel),
        },
        OUT / "decoupled18k.pt",
    )
    print("scoring official-ish test (our 13 segments, 100 Hz)", flush=True)
    paper = score(vel, att, test)
    print(
        f"\nours 18k decoupled   delta_z {paper['delta_z']:.4f}  "
        f"delta_q {paper['delta_q']:.4f} rad  windows {paper['windows']}"
    )
    print("paper TCN 5.2M×2 decoupled   delta_z 0.042   delta_q 0.006")
    print("paper GRU 5.2M×2 decoupled   delta_z 0.090   delta_q 0.010")
    print(f"wrote {OUT / 'decoupled18k.pt'}")


if __name__ == "__main__":
    main()
