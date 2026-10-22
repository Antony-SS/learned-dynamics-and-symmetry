"""Train one hist-10 Z2×Z2 with stride-5 cached windows; compare to headline.

Does not overwrite runs/idsia_mlp_hist10/hist10_parity.pt or idsia_parity18k.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quadrotor.fit import n_params, train_device
from quadrotor.idsia.data import load_idsia_prepared
from quadrotor.idsia.history_mlp import (
    HEADLINE_DIR,
    TRAIN_STRIDE,
    load_or_train_headline,
    row,
    save_model,
    score_history,
    train_history,
)

OUT = ROOT / "runs" / "idsia_mlp_hist10_stride5"


def main() -> None:
    device = train_device()
    train, _val, test = load_idsia_prepared()
    print(
        f"one Z2×Z2  train stride {TRAIN_STRIDE} (jump {TRAIN_STRIDE} ticks / "
        f"{10 * TRAIN_STRIDE} ms)  Melon still stride 1  device {device}",
        flush=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "hist10_parity_stride5.pt"
    if ckpt.is_file():
        from quadrotor.idsia.history_mlp import load_model

        fast = load_model(ckpt, device)
        print(f"loaded {ckpt}", flush=True)
    else:
        fast = train_history(
            train,
            history=10,
            symmetric=True,
            device=device,
            seed=0,
            tag="hist-10 Z2×Z2 stride-5",
            stride=TRAIN_STRIDE,
        )
        save_model(fast, ckpt)
        print(f"saved {ckpt}", flush=True)

    old = load_or_train_headline(device=device)
    print(f"headline {HEADLINE_DIR / 'hist10_parity.pt'}  params {n_params(old)}", flush=True)
    print(f"stride-5  params {n_params(fast)}", flush=True)
    print("\nMelon MAE  h=10 / h=50  (stride-1 windows)", flush=True)
    row(f"headline stride-1 ({n_params(old)})", score_history(old, test, device))
    row(f"this run stride-5 ({n_params(fast)})", score_history(fast, test, device))


if __name__ == "__main__":
    main()
