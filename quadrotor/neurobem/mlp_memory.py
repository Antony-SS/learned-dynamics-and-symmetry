"""Does history or explicit products raise the 64-64 MLP α R²?"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from quadrotor.neurobem.data import DEFAULT_ROOT, _flight_id, _segment_id

OUT = Path(__file__).resolve().parent / "out" / "observability"
HIDDEN = 64
EPOCHS = 25
BATCH = 4096


def _split(root: Path, seed: int = 0) -> tuple[list[Path], list[Path]]:
    paths = sorted((root / "processed_data").glob("merged_*.csv"))
    test_ids = {
        line.strip()
        for line in (root / "testset.txt").read_text().splitlines()
        if line.strip()
    }
    test = [path for path in paths if _segment_id(path) in test_ids]
    remaining = [path for path in paths if _segment_id(path) not in test_ids]
    by_flight: dict[str, list[Path]] = {}
    for path in remaining:
        by_flight.setdefault(_flight_id(path), []).append(path)
    flight_ids = sorted(by_flight)
    shuffled = [
        flight_ids[i] for i in np.random.default_rng(seed).permutation(len(flight_ids))
    ]
    n_val = max(1, int(round(0.20 * len(shuffled))))
    train = [path for flight in shuffled[n_val:] for path in sorted(by_flight[flight])]
    return train, test


def _load(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    return {
        "omega": frame[["ang vel x", "ang vel y", "ang vel z"]].to_numpy(np.float32),
        "alpha": frame[["ang acc x", "ang acc y", "ang acc z"]].to_numpy(np.float32),
        "vel": frame[["vel x", "vel y", "vel z"]].to_numpy(np.float32),
        "motors": frame[["mot 2", "mot 1", "mot 3", "mot 4"]].to_numpy(np.float32),
    }


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    residual = y - yhat
    denom = ((y - y.mean(0)) ** 2).sum()
    return float(1.0 - (residual ** 2).sum() / max(denom, 1e-18))


def _axis_r2(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return {
        "x": _r2(y[:, 0:1], yhat[:, 0:1]),
        "y": _r2(y[:, 1:2], yhat[:, 1:2]),
        "z": _r2(y[:, 2:3], yhat[:, 2:3]),
        "all": _r2(y, yhat),
    }


def quotient(vel: np.ndarray, omega: np.ndarray, motors: np.ndarray) -> np.ndarray:
    return np.concatenate([vel, omega, motors ** 2], axis=1)


def products(vel: np.ndarray, motors: np.ndarray) -> np.ndarray:
    return (vel[:, :, None] * motors[:, None, :]).reshape(len(vel), 12)


def history_stack(flights: list[dict], k: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for flight in flights:
        q = quotient(flight["vel"], flight["omega"], flight["motors"])
        if len(q) < k:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(q, (k, q.shape[1]))[:, 0]
        xs.append(np.ascontiguousarray(windows.reshape(len(windows), -1)))
        ys.append(flight["alpha"][k - 1 :])
    return np.concatenate(xs), np.concatenate(ys)


class RateMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedMLP(torch.nn.Module):
    """Two-branch product: (v,ω) × Ω², then the same 64-64 head."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.state = torch.nn.Linear(6, hidden)
        self.motors = torch.nn.Linear(4, hidden)
        self.head = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.state(x[..., :6]) * self.motors(x[..., 6:]))


def fit(
    name: str,
    model: torch.nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float]:
    x_mean, x_std = train_x.mean(0), np.clip(train_x.std(0), 1e-6, None)
    y_mean, y_std = train_y.mean(0), np.clip(train_y.std(0), 1e-6, None)
    xt = torch.as_tensor((train_x - x_mean) / x_std, dtype=torch.float32)
    yt = torch.as_tensor((train_y - y_mean) / y_std, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"{name:28s}  in {train_x.shape[1]:3d}  params {n_param}  n {len(train_x)}", flush=True)
    for epoch in range(EPOCHS):
        order = torch.randperm(len(xt))
        total = 0.0
        model.train()
        for start in range(0, len(order), BATCH):
            idx = order[start : start + BATCH]
            pred = model(xt[idx])
            loss = (pred - yt[idx]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  {name:28s}  epoch {epoch:02d}  train {total / len(xt):.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        xe = torch.as_tensor((test_x - x_mean) / x_std, dtype=torch.float32)
        hat = model(xe).numpy() * y_std + y_mean
    scores = _axis_r2(test_y, hat)
    print(
        f"  {name:28s}  test R²  all {scores['all']:.3f}  "
        f"x {scores['x']:.3f}  y {scores['y']:.3f}  z {scores['z']:.3f}",
        flush=True,
    )
    return scores


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_paths, test_paths = _split(DEFAULT_ROOT)
    print(f"loading 400 Hz  train {len(train_paths)}  test {len(test_paths)}", flush=True)
    train = [_load(p) for p in train_paths]
    test = [_load(p) for p in test_paths]

    tr_v = np.concatenate([f["vel"] for f in train])
    tr_w = np.concatenate([f["omega"] for f in train])
    tr_m = np.concatenate([f["motors"] for f in train])
    tr_a = np.concatenate([f["alpha"] for f in train])
    te_v = np.concatenate([f["vel"] for f in test])
    te_w = np.concatenate([f["omega"] for f in test])
    te_m = np.concatenate([f["motors"] for f in test])
    te_a = np.concatenate([f["alpha"] for f in test])
    q_tr, q_te = quotient(tr_v, tr_w, tr_m), quotient(te_v, te_w, te_m)
    prod_tr = np.concatenate([q_tr, products(tr_v, tr_m)], axis=1)
    prod_te = np.concatenate([q_te, products(te_v, te_m)], axis=1)

    results: dict[str, dict[str, float]] = {}
    torch.manual_seed(0)
    results["MLP (v,ω,Ω²)"] = fit("MLP (v,ω,Ω²)", RateMLP(10), q_tr, tr_a, q_te, te_a)
    torch.manual_seed(0)
    results["MLP + v⊗Ω"] = fit("MLP + v⊗Ω", RateMLP(22), prod_tr, tr_a, prod_te, te_a)
    torch.manual_seed(0)
    results["gated (v,ω)×Ω²"] = fit("gated (v,ω)×Ω²", GatedMLP(), q_tr, tr_a, q_te, te_a)

    for k in (5, 10, 20):
        x_tr, y_tr = history_stack(train, k)
        x_te, y_te = history_stack(test, k)
        torch.manual_seed(0)
        results[f"MLP history {k}"] = fit(
            f"MLP history {k} ({1000.0 * k / 400.0:.0f} ms)",
            RateMLP(10 * k),
            x_tr,
            y_tr,
            x_te,
            y_te,
        )

    print("\n==== official-test α R² ====")
    print(f"{'model':28s}  {'all':>6}  {'x':>6}  {'y':>6}  {'z':>6}")
    for name, scores in results.items():
        print(
            f"{name:28s}  {scores['all']:6.3f}  {scores['x']:6.3f}  "
            f"{scores['y']:6.3f}  {scores['z']:6.3f}"
        )
    np.savez(
        OUT / "mlp_memory_r2.npz",
        names=np.asarray(list(results)),
        all=np.asarray([s["all"] for s in results.values()]),
        x=np.asarray([s["x"] for s in results.values()]),
        y=np.asarray([s["y"] for s in results.values()]),
        z=np.asarray([s["z"] for s in results.values()]),
    )

    names = list(results)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(names))
    width = 0.2
    for i, axis in enumerate(("all", "x", "y", "z")):
        ax.bar(
            x + (i - 1.5) * width,
            [results[n][axis] for n in names],
            width,
            label=axis,
        )
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("test α R²")
    ax.set_ylim(0, 1)
    ax.set_title("NeuroBEM 400 Hz: memory and products vs the 64-64 MLP")
    ax.legend(frameon=False, ncol=4)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "mlp_memory_r2.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'mlp_memory_r2.png'}")


if __name__ == "__main__":
    main()
