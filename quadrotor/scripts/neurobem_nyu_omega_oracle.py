"""Zero-parameter ω baseline on the NYU decoupled Δz clock.

ω̂_t = Log(R_tᵀ R_{t+1}) / Δt   (forward; uses next attitude)
ω̂_t = Log(R_{t-1}ᵀ R_t) / Δt   (backward; attitude only up to t)

No training. Same 100 Hz test windows as the 18k decoupled score.
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
from quadrotor.neurobem.data import load_neurobem_prepared
from quadrotor.train import _so3_log


def _phi(r0: torch.Tensor, r1: torch.Tensor) -> torch.Tensor:
    return _so3_log(torch.matmul(r0.transpose(-1, -2), r1))


@torch.no_grad()
def score(flights) -> dict[str, float]:
    pairs = nyu.windows(flights, nyu.EVAL_H, nyu.EVAL_STRIDE)
    device = torch.device("cpu")
    acc = {k: 0.0 for k in ("fwd_t", "fwd_tp1", "fwd_mid", "bwd_t")}
    n = 0
    for offset in range(0, len(pairs), nyu.BATCH):
        batch = pairs[offset : offset + nyu.BATCH]
        _pwm, dt, _p, r, _v, w = nyu.tensors(flights, batch, nyu.EVAL_H, device)
        # states are length H+1; intervals length H; dt is length H
        phi_fwd = _phi(r[:, :-1], r[:, 1:])
        w_fwd = phi_fwd / dt.unsqueeze(-1).clamp_min(1e-6)
        phi_bwd = _phi(r[:, :-1], r[:, 1:])
        # backward at step t+1 uses R_t → R_{t+1}, already phi_fwd; align to w[:, 1:]
        w_bwd = phi_bwd / dt.unsqueeze(-1).clamp_min(1e-6)
        acc["fwd_t"] += float((w_fwd - w[:, :-1]).square().sum(-1).sum())
        acc["fwd_tp1"] += float((w_fwd - w[:, 1:]).square().sum(-1).sum())
        acc["fwd_mid"] += float((w_fwd - 0.5 * (w[:, :-1] + w[:, 1:])).square().sum(-1).sum())
        acc["bwd_t"] += float((w_bwd - w[:, 1:]).square().sum(-1).sum())
        n += w_fwd.shape[0] * w_fwd.shape[1]
    return {k: v / max(n, 1) for k, v in acc.items()} | {"windows": len(pairs), "n": n}


def main() -> None:
    _train, _val, raw_test = load_neurobem_prepared()
    test = nyu.keep(raw_test, nyu.EVAL_H)
    out = score(test)
    print(
        f"zero-param ω  100 Hz  T={nyu.EVAL_H}  stride {nyu.EVAL_STRIDE}  "
        f"windows {out['windows']}  steps {out['n']:,}",
        flush=True,
    )
    print(f"  fwd  Log(R_tᵀR_{{t+1}})/Δt vs ω_t       Δz_ω {out['fwd_t']:.4f}")
    print(f"  fwd  Log(R_tᵀR_{{t+1}})/Δt vs ω_{{t+1}}    Δz_ω {out['fwd_tp1']:.4f}")
    print(f"  fwd  vs mid-step (ω_t+ω_{{t+1}})/2        Δz_ω {out['fwd_mid']:.4f}")
    print(f"  bwd  Log(R_tᵀR_{{t+1}})/Δt vs ω_{{t+1}}    Δz_ω {out['bwd_t']:.4f}  (attitude up to t+1)")
    print("  ours 18k decoupled                       Δz_ω 0.8558")
    print("  paper TCN / GRU full Δz                  0.042 / 0.090")


if __name__ == "__main__":
    main()
