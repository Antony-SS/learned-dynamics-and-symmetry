"""Rao et al. (IROS 2024) clock, open-loop (coupled) NeuroBEM test.

Paper: 100 Hz, 20-step warm (0.2 s), 60-step unroll (0.6 s).
delta_z = mean_t(||v-vhat||^2 + ||w-what||^2)
delta_q = mean_t geodesic attitude [rad]

Their published TCN/GRU numbers are the *decoupled* protocol, not this.
This script is the coupled / open-loop version on that same horizon.
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

from quadrotor.evaluate import benchmark_mae
from quadrotor.fit import SymmetricModel, unpack_state
from quadrotor.neurobem.data import load_neurobem_prepared
from quadrotor.train import _integrate, _so3_log, module_device, prime_hidden

CKPT = ROOT / "runs" / "neurobem_parity18k" / "symmetric.pt"
EVERY = 4  # 400 Hz -> 100 Hz
HISTORY = 20
HORIZON = 60
STRIDE = 10  # 0.10 s between seeds; paper used 1
BATCH = 64
PAPER = {
    "tcn_decoupled": (0.042, 0.006),
    "gru_decoupled": (0.090, 0.010),
}


def at_100hz(flight):
    n = (flight.n_steps // EVERY) * EVERY
    if n < EVERY * (HISTORY + HORIZON):
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


@torch.no_grad()
def paper_metrics(model, flights) -> dict[str, float]:
    pairs = [
        (i, s)
        for i, f in enumerate(flights)
        for s in range(HISTORY, f.n_steps - HORIZON + 1, STRIDE)
    ]
    device = module_device(model)
    dz = 0.0
    dq = 0.0
    seen = 0
    model.eval()
    for offset in range(0, len(pairs), BATCH):
        batch = pairs[offset : offset + BATCH]
        pwm, dt, p, r, v, w = [], [], [], [], [], []
        for fi, s in batch:
            f = flights[fi]
            stop = s + HORIZON
            pwm.append(f.pwm[s:stop])
            dt.append(f.dt[s:stop])
            p.append(f.position[s : stop + 1])
            r.append(f.rotation[s : stop + 1])
            v.append(f.velocity_world[s : stop + 1])
            w.append(f.omega_body[s : stop + 1])
        pwm_t = torch.as_tensor(np.stack(pwm), device=device)
        dt_t = torch.as_tensor(np.stack(dt), device=device)
        p_t = torch.as_tensor(np.stack(p), device=device)
        r_t = torch.as_tensor(np.stack(r), device=device)
        v_t = torch.as_tensor(np.stack(v), device=device)
        w_t = torch.as_tensor(np.stack(w), device=device)
        hidden = prime_hidden(model, flights, batch, HISTORY)
        p_hat, r_hat, v_hat, w_hat = _integrate(
            model, pwm_t, dt_t, p_t[:, 0], r_t[:, 0], v_t[:, 0], w_t[:, 0], None, hidden=hidden
        )
        rel = torch.matmul(r_t[:, 1:].transpose(-1, -2), r_hat)
        vel_sq = (v_hat - v_t[:, 1:]).square().sum(-1) + (w_hat - w_t[:, 1:]).square().sum(-1)
        att = torch.linalg.vector_norm(_so3_log(rel.reshape(-1, 3, 3)), dim=-1)
        dz += float(vel_sq.sum().cpu())
        dq += float(att.sum().cpu())
        seen += vel_sq.numel()
        print(f"  windows {min(offset + BATCH, len(pairs))}/{len(pairs)}", flush=True)
    return {
        "delta_z": dz / max(seen, 1),
        "delta_q": dq / max(seen, 1),
        "windows": len(pairs),
    }


def main() -> None:
    if not CKPT.is_file():
        raise SystemExit(f"missing {CKPT}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _, _, test = load_neurobem_prepared()
    test = [f for f in (at_100hz(f) for f in test) if f.n_steps > HISTORY + HORIZON]
    print(
        f"NYU clock  100 Hz  warm {HISTORY} (0.20 s)  unroll {HORIZON} (0.60 s)  "
        f"open-loop / coupled  test flights {len(test)}",
        flush=True,
    )
    model = SymmetricModel(include_voltage=False)
    model.load_state_dict(unpack_state(torch.load(CKPT, map_location="cpu", weights_only=False)))
    model.to(device).eval()
    print(f"loaded {CKPT}  scoring paper metrics", flush=True)
    paper = paper_metrics(model, test)
    print("scoring end-of-horizon MAE", flush=True)
    mae = benchmark_mae(
        model,
        test,
        horizon=HORIZON,
        start_stride=STRIDE,
        start_offset=HISTORY,
        warm_steps=HISTORY,
    )
    print(
        f"\nours (coupled, 0.6 s mean)   delta_z {paper['delta_z']:.4f}  "
        f"delta_q {paper['delta_q']:.4f} rad  windows {paper['windows']}"
    )
    print(
        f"ours (coupled, 0.6 s end)    p {mae['position'][-1]:.4f}  "
        f"R {mae['rotation'][-1]:.4f}  v {mae['velocity'][-1]:.4f}  "
        f"w {mae['omega'][-1]:.4f}"
    )
    print("paper (decoupled, not open-loop)  TCN  dz 0.042  dq 0.006")
    print("paper (decoupled, not open-loop)  GRU  dz 0.090  dq 0.010")


if __name__ == "__main__":
    main()
