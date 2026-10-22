"""Quotient parity MLP, full SE(3), train loss on 200 steps (0.5 s) like the GRU."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quadrotor.scripts.neurobem_mlp_fullstate as fs  # noqa: E402

NAME = "quotient-parity-h200"
TRAIN_H = 200


def main() -> None:
    fs.TRAIN_H = TRAIN_H
    fs.OUT.mkdir(parents=True, exist_ok=True)
    device = fs.torch.device("cpu")
    train_paths, _val, test_paths = fs._split(fs.DEFAULT_ROOT)
    print(f"loading 400 Hz  train rollout {TRAIN_H} steps (0.50 s)", flush=True)
    train = [fs._load(p) for p in train_paths]
    test = [fs._load(p) for p in test_paths]
    pairs = fs._windows(train, TRAIN_H, fs.STRIDE)
    scales = fs._state_scales(train)
    mean, scale = fs._feature_stats(train)
    mean, scale = mean.to(device), scale.to(device)
    print(
        f"train {len(train)}  test {len(test)}  windows {len(pairs)}  "
        f"epochs {fs.EPOCHS}  batch {fs.BATCH}",
        flush=True,
    )
    fs.torch.manual_seed(0)
    model = fs.AccelMLP(13, symmetry="parity")
    fs.train_one(
        NAME,
        model,
        train,
        pairs,
        device=device,
        x_mean=mean,
        x_scale=scale,
        scales=scales,
    )
    fs.torch.save(
        {
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "symmetry": "parity",
            "train_h": TRAIN_H,
            "x_mean": mean.cpu(),
            "x_scale": scale.cpu(),
        },
        fs.OUT / f"fullstate_{NAME}.pt",
    )
    print("scoring 0.5 s official-test rollouts", flush=True)
    coast = fs.score_model(None, test, device=device, coast=True)
    pred = fs.score_model(model, test, device=device, x_mean=mean, x_scale=scale)
    t = np.arange(1, fs.EVAL_H + 1) / fs.RATE_HZ
    print(f"\n{'model':22s}  {'p':>8}  {'R':>8}  {'v':>8}  {'ω':>8}  (h=200)")
    for label, curve in (("coast", coast), (NAME, pred)):
        print(
            f"{label:22s}  {curve['position'][199]:8.4f}  {curve['rotation'][199]:8.4f}  "
            f"{curve['velocity'][199]:8.4f}  {curve['omega'][199]:8.4f}"
        )
    print("\nreference (parity train h=50):  ω h200 ~1.358")
    print("reference (GRU val ep5):        ω h200 ~0.574 vs coast ~1.465")
    np.savez(
        fs.OUT / f"fullstate_{NAME}_curves.npz",
        t=t,
        **{f"{k}_{NAME}": pred[k] for k in pred},
        **{f"{k}_coast": coast[k] for k in coast},
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(t, coast["omega"], "k--", lw=2.2, label="coast")
    ax.plot(t, pred["omega"], color="#2a9d8f", lw=2, label=f"{NAME} ω")
    ax.set_xlabel("horizon [s]")
    ax.set_ylabel("ω MAE [rad/s]")
    ax.set_title("Full SE(3) parity MLP, train loss on 200 steps")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    fig.tight_layout()
    fig.savefig(fs.OUT / f"fullstate_{NAME}_omega.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fs.OUT / f'fullstate_{NAME}_omega.png'}")


if __name__ == "__main__":
    main()
