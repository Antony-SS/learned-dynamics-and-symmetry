"""Split Δz, finite-diff ω oracle, then increment-input velocity net.

Does not touch runs/neurobem_parity18k. Writes
runs/neurobem_nyu_decoupled18k/increment.pt only.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quadrotor.scripts.neurobem_nyu_decoupled18k as nyu
from quadrotor.fit import SymmetricModel, n_params, train_device, unpack_state
from quadrotor.neurobem.data import load_neurobem_prepared
from quadrotor.train import (
    _configure_input_normalization,
    _features,
    _so3_log,
    module_device,
    prime_hidden,
)

CKPT = nyu.OUT / "decoupled18k.pt"


class IncrementParity(torch.nn.Module):
    """Same 18k leftover plus a tiny map from Log(R_t^T R_{t+1})."""

    reads_pose = False
    mirror = True

    def __init__(self) -> None:
        super().__init__()
        self.core = SymmetricModel(include_voltage=False)
        self.inc = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 6),
        )
        torch.nn.init.zeros_(self.inc[-1].weight)
        torch.nn.init.zeros_(self.inc[-1].bias)

    def initial_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return self.core.initial_hidden(batch_size, reference)

    def set_input_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        self.core.set_input_normalization(center, scale)

    def step(self, features: torch.Tensor, hidden: torch.Tensor, increment: torch.Tensor):
        acc, hidden = self.core.step(features, hidden)
        return acc + self.inc(increment), hidden


def increments(r: torch.Tensor, horizon: int) -> torch.Tensor:
    rel = torch.matmul(r[:, :horizon].transpose(-1, -2), r[:, 1 : horizon + 1])
    return _so3_log(rel.reshape(-1, 3, 3)).reshape(r.shape[0], horizon, 3)


def roll_velocity_inc(model, pwm, dt, r_gt, v, w, hidden, inc):
    vs, ws = [], []
    dummy = v.new_zeros(v.shape[0], 3)
    for t in range(pwm.shape[1]):
        rot = r_gt[:, t]
        acc, hidden = model.step(
            _features(model, dummy, rot, v, w, pwm[:, t]), hidden, inc[:, t]
        )
        step = dt[:, t].unsqueeze(-1)
        v = v + torch.bmm(rot, acc[:, 0:3].unsqueeze(-1)).squeeze(-1) * step
        w = w + acc[:, 3:6] * step
        vs.append(v)
        ws.append(w)
    return torch.stack(vs, 1), torch.stack(ws, 1)


def prime_inc(model, flights, pairs, device):
    hidden = model.initial_hidden(len(pairs), torch.zeros(len(pairs), 1, device=device))
    dummy = torch.zeros(len(pairs), 3, device=device)
    for offset in range(nyu.HISTORY):
        pos, rot, vel, ome, pwm, nxt = [], [], [], [], [], []
        for index, start in pairs:
            flight = flights[index]
            step = min(max(0, start - nyu.HISTORY + offset), flight.n_steps - 2)
            pos.append(flight.position[step])
            rot.append(flight.rotation[step])
            nxt.append(flight.rotation[step + 1])
            vel.append(flight.velocity_world[step])
            ome.append(flight.omega_body[step])
            pwm.append(flight.pwm[step])
        r0 = torch.as_tensor(np.stack(rot), device=device)
        r1 = torch.as_tensor(np.stack(nxt), device=device)
        inc = _so3_log(torch.matmul(r0.transpose(1, 2), r1))
        feat = _features(
            model,
            dummy,
            r0,
            torch.as_tensor(np.stack(vel), device=device),
            torch.as_tensor(np.stack(ome), device=device),
            torch.as_tensor(np.stack(pwm), device=device),
        )
        _, hidden = model.step(feat, hidden, inc)
    return hidden


@torch.no_grad()
def score_split(vel, flights, use_increment: bool = False) -> dict[str, float]:
    pairs = nyu.windows(flights, nyu.EVAL_H, nyu.EVAL_STRIDE)
    device = module_device(vel)
    dv = dw = n = 0.0
    vel.eval()
    for offset in range(0, len(pairs), nyu.BATCH):
        batch = pairs[offset : offset + nyu.BATCH]
        pwm, dt, _p, r, v, w = nyu.tensors(flights, batch, nyu.EVAL_H, device)
        if use_increment:
            hidden = prime_inc(vel, flights, batch, device)
            v_hat, w_hat = roll_velocity_inc(
                vel, pwm, dt, r, v[:, 0], w[:, 0], hidden, increments(r, nyu.EVAL_H)
            )
        else:
            hidden = prime_hidden(vel, flights, batch, nyu.HISTORY)
            v_hat, w_hat = nyu.roll_velocity(vel, pwm, dt, r, v[:, 0], w[:, 0], hidden)
        dv += float((v_hat - v[:, 1:]).square().sum(-1).sum().cpu())
        dw += float((w_hat - w[:, 1:]).square().sum(-1).sum().cpu())
        n += float((v_hat - v[:, 1:]).square().sum(-1).numel())
    return {
        "delta_v": dv / n,
        "delta_w": dw / n,
        "delta_z": (dv + dw) / n,
        "windows": len(pairs),
    }


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.square(target - pred).sum())
    ss_tot = float(np.square(target - target.mean(axis=0)).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def oracle_omega(flights) -> dict[str, float]:
    fd, logged, mid = [], [], []
    for flight in flights:
        rel = np.matmul(
            np.swapaxes(flight.rotation[:-1], -1, -2), flight.rotation[1:]
        )
        phi = _so3_log(torch.as_tensor(rel)).numpy()
        w_fd = phi / np.clip(flight.dt[:, None], 1e-6, None)
        w0 = flight.omega_body[:-1]
        w1 = flight.omega_body[1:]
        fd.append(w_fd)
        logged.append(w0)
        mid.append(0.5 * (w0 + w1))
    fd = np.concatenate(fd)
    logged = np.concatenate(logged)
    mid = np.concatenate(mid)
    err = fd - logged
    err_mid = fd - mid
    return {
        "rmse_vs_logged": float(np.sqrt(np.mean(np.square(err)))),
        "mae_vs_logged": float(np.mean(np.linalg.norm(err, axis=1))),
        "r2_vs_logged": _r2(fd, logged),
        "corr_vs_logged": float(
            np.corrcoef(fd.reshape(-1), logged.reshape(-1))[0, 1]
        ),
        "rmse_vs_mid": float(np.sqrt(np.mean(np.square(err_mid)))),
        "r2_vs_mid": _r2(fd, mid),
        "mean_sq_w": float(np.mean(np.sum(np.square(logged), axis=1))),
        "mean_sq_fd": float(np.mean(np.sum(np.square(fd), axis=1))),
        "n": int(len(fd)),
    }


def train_increment(model, flights, pairs) -> None:
    device = module_device(model)
    opt = torch.optim.Adam(model.parameters(), lr=nyu.LR)
    best = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for epoch in range(1, nyu.EPOCHS + 1):
        order = torch.randperm(len(pairs)).tolist()
        model.train()
        total = seen = 0.0
        for offset in range(0, len(order), nyu.BATCH):
            batch = [pairs[i] for i in order[offset : offset + nyu.BATCH]]
            pwm, dt, _p, r, v, w = nyu.tensors(flights, batch, nyu.TRAIN_H, device)
            hidden = prime_inc(model, flights, batch, device)
            v_hat, w_hat = roll_velocity_inc(
                model, pwm, dt, r, v[:, 0], w[:, 0], hidden, increments(r, nyu.TRAIN_H)
            )
            loss = (v_hat - v[:, 1:]).square().mean() + (w_hat - w[:, 1:]).square().mean()
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
        print(f"inc  epoch {epoch:02d}/{nyu.EPOCHS}  train {metric:.5f}  best {best:.5f}", flush=True)
    model.load_state_dict(best_state)
    model.to(device)


def main() -> None:
    device = train_device()
    raw_train, _val, raw_test = load_neurobem_prepared()
    train, test = nyu.keep(raw_train, nyu.EVAL_H), nyu.keep(raw_test, nyu.EVAL_H)
    print(f"100 Hz  test flights {len(test)}  device {device}", flush=True)

    if not CKPT.is_file():
        raise SystemExit(f"missing {CKPT}")
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    vel = SymmetricModel(include_voltage=False)
    vel.load_state_dict(unpack_state(payload["vel"]))
    vel.to(device).eval()

    print("1) split Δz on saved 18k decoupled vel net", flush=True)
    split = score_split(vel, test, use_increment=False)
    print(
        f"   delta_z {split['delta_z']:.4f}  =  delta_v {split['delta_v']:.4f}  "
        f"+ delta_w {split['delta_w']:.4f}   ({100 * split['delta_w'] / max(split['delta_z'], 1e-12):.1f}% ω)  "
        f"windows {split['windows']}",
        flush=True,
    )

    print("2) finite-diff ω from logged R vs logged ω", flush=True)
    for name, flights in (("train", train), ("test", test)):
        ora = oracle_omega(flights)
        print(
            f"   {name:5s}  n {ora['n']:,}  RMSE {ora['rmse_vs_logged']:.4f}  "
            f"MAE {ora['mae_vs_logged']:.4f}  R² {ora['r2_vs_logged']:.3f}  "
            f"corr {ora['corr_vs_logged']:.3f}  vs mid-step R² {ora['r2_vs_mid']:.3f}  "
            f"mean||ω||² {ora['mean_sq_w']:.3f}  mean||ω_fd||² {ora['mean_sq_fd']:.3f}",
            flush=True,
        )

    print("3) train 18k leftover + Log(R_tᵀ R_{t+1}) input", flush=True)
    model = IncrementParity().to(device)
    _configure_input_normalization(model.core, train)
    print(f"   params {n_params(model):,}", flush=True)
    train_increment(model, train, nyu.windows(train, nyu.TRAIN_H, nyu.TRAIN_STRIDE))
    nyu.OUT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
        nyu.OUT / "increment.pt",
    )
    inc = score_split(model, test, use_increment=True)
    print(
        f"\n   increment-input  delta_z {inc['delta_z']:.4f}  =  "
        f"delta_v {inc['delta_v']:.4f}  + delta_w {inc['delta_w']:.4f}",
        flush=True,
    )
    print(f"   baseline 18k     delta_z {split['delta_z']:.4f}  =  "
          f"delta_v {split['delta_v']:.4f}  + delta_w {split['delta_w']:.4f}")
    print("   paper GRU / TCN  delta_z 0.090 / 0.042")


if __name__ == "__main__":
    main()
