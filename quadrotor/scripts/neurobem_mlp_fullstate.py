"""World vs quotient 64-64, ± LR, full SE(3) rollout. Pose via g ← g ∘ Exp(Δt ν)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from quadrotor.geometry import quat_to_matrix
from quadrotor.neurobem.data import DEFAULT_ROOT, _flight_id, _segment_id
from quadrotor.sequence import _mirror_acceleration, _mirror_features
from quadrotor.train import _so3_exp, _so3_log

RATE_HZ = 400.0
EVAL_H = 200
TRAIN_H = 50
STRIDE = 40
HIDDEN = 64
EPOCHS = 15
BATCH = 256
OUT = Path("runs/neurobem_observability")


def _split(root: Path, seed: int = 0) -> tuple[list[Path], list[Path], list[Path]]:
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
    val = [path for flight in shuffled[:n_val] for path in sorted(by_flight[flight])]
    train = [path for flight in shuffled[n_val:] for path in sorted(by_flight[flight])]
    return train, val, test


def _load(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    rotation = quat_to_matrix(
        frame[["quat x", "quat y", "quat z", "quat w"]].to_numpy(np.float64)
    ).astype(np.float32)
    vel_body = frame[["vel x", "vel y", "vel z"]].to_numpy(np.float32)
    return {
        "position": frame[["pos x", "pos y", "pos z"]].to_numpy(np.float32),
        "rotation": rotation,
        "vel_body": vel_body,
        "vel_world": np.einsum("nij,nj->ni", rotation, vel_body).astype(np.float32),
        "omega": frame[["ang vel x", "ang vel y", "ang vel z"]].to_numpy(np.float32),
        "motors": frame[["mot 2", "mot 1", "mot 3", "mot 4"]].to_numpy(np.float32),
        "dt": np.diff(frame["t"].to_numpy(np.float64)).astype(np.float32),
    }


def _windows(flights: list[dict], horizon: int, stride: int) -> list[tuple[int, int]]:
    return [
        (i, start)
        for i, flight in enumerate(flights)
        for start in range(0, len(flight["dt"]) - horizon, stride)
    ]


def _hat(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zeros, -z, y), dim=-1),
            torch.stack((z, zeros, -x), dim=-1),
            torch.stack((-y, x, zeros), dim=-1),
        ),
        dim=-2,
    )


def _se3_exp_update(
    position: torch.Tensor,
    rotation: torch.Tensor,
    vel_body: torch.Tensor,
    omega: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """g ← g ∘ Exp(Δt ν) with body twist ν = (v, ω)."""
    dt = dt.unsqueeze(-1)
    phi = omega * dt
    increment = _so3_exp(phi)
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True)
    skew = _hat(phi)
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(phi.shape[0], 3, 3)
    small = theta < 1e-8
    theta_safe = theta.clamp_min(1e-8)
    a = (1.0 - torch.cos(theta_safe)) / theta_safe.square()
    b = (theta_safe - torch.sin(theta_safe)) / (theta_safe.square() * theta_safe)
    left = eye + a[..., None] * skew + b[..., None] * (skew @ skew)
    left = torch.where(small[..., None], eye + 0.5 * skew, left)
    disp = torch.bmm(left, (vel_body * dt).unsqueeze(-1)).squeeze(-1)
    position = position + torch.bmm(rotation, disp.unsqueeze(-1)).squeeze(-1)
    rotation = torch.bmm(rotation, increment)
    return position, rotation


def quotient_features(
    vel_body: torch.Tensor, omega: torch.Tensor, rotation: torch.Tensor, motors: torch.Tensor
) -> torch.Tensor:
    """Same 13-D canonical input as MotorNet: (Rᵀe₃, v_body, ω, Ω)."""
    up = rotation.new_tensor((0.0, 0.0, 1.0)).expand(rotation.shape[0], 3)
    gravity = torch.bmm(rotation.transpose(1, 2), up.unsqueeze(-1)).squeeze(-1)
    return torch.cat((gravity, vel_body, omega, motors), dim=-1)


class AccelMLP(torch.nn.Module):
    """Feedforward map on 13-D quotient features → body (a, α)."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = HIDDEN,
        *,
        symmetry: str | None = None,
    ) -> None:
        super().__init__()
        if symmetry not in (None, "lr", "parity"):
            raise ValueError(f"symmetry must be None, 'lr', or 'parity', got {symmetry!r}")
        self.symmetry = symmetry
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 6),
        )

    def raw(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def _orbit_average(self, features: torch.Tensor) -> torch.Tensor:
        """One batched MLP pass over the mirror orbit (same math as four forwards)."""
        batch = features.shape[0]
        mirrors = (None, "lr", "fb", "both")
        stacked = torch.cat(
            [
                features if name is None else _mirror_features(features, name)
                for name in mirrors
            ],
            dim=0,
        )
        raw = self.raw(stacked).reshape(len(mirrors), batch, 6)
        total = raw[0]
        for index, name in enumerate(mirrors[1:], start=1):
            total = total + _mirror_acceleration(raw[index], name)
        return total / float(len(mirrors))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.symmetry is None:
            return self.raw(features)
        if self.symmetry == "lr":
            batch = features.shape[0]
            stacked = torch.cat(
                (features, _mirror_features(features, "lr")),
                dim=0,
            )
            raw = self.raw(stacked).reshape(2, batch, 6)
            return 0.5 * (raw[0] + _mirror_acceleration(raw[1], "lr"))
        return self._orbit_average(features)


def _state_scales(flights: list[dict]) -> tuple[float, float, float, float]:
    position = np.concatenate([f["position"] for f in flights])
    velocity = np.concatenate([f["vel_world"] for f in flights])
    omega = np.concatenate([f["omega"] for f in flights])
    rotation = np.concatenate([f["rotation"] for f in flights])
    rotvec = _so3_log(torch.as_tensor(rotation)).numpy()
    return (
        max(float(np.std(position)), 1e-6),
        max(float(np.std(rotvec)), 1e-6),
        max(float(np.std(velocity)), 1e-6),
        max(float(np.std(omega)), 1e-6),
    )


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


def _features_for(
    rotation: torch.Tensor,
    vel_body: torch.Tensor,
    omega: torch.Tensor,
    motors: torch.Tensor,
) -> torch.Tensor:
    return quotient_features(vel_body, omega, rotation, motors)


def _roll(
    model: AccelMLP,
    *,
    position: torch.Tensor,
    rotation: torch.Tensor,
    vel_body: torch.Tensor,
    omega: torch.Tensor,
    motors: torch.Tensor,
    dt: torch.Tensor,
    x_mean: torch.Tensor,
    x_scale: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ps, rs, vs, ws = [], [], [], []
    p, r, v, w = position, rotation, vel_body, omega
    for k in range(horizon):
        feat = (_features_for(r, v, w, motors[:, k]) - x_mean) / x_scale
        acc = model(feat)
        v = v + acc[:, 0:3] * dt[:, k].unsqueeze(-1)
        w = w + acc[:, 3:6] * dt[:, k].unsqueeze(-1)
        p, r = _se3_exp_update(p, r, v, w, dt[:, k])
        ps.append(p)
        rs.append(r)
        vs.append(torch.bmm(r, v.unsqueeze(-1)).squeeze(-1))
        ws.append(w)
    return torch.stack(ps, 1), torch.stack(rs, 1), torch.stack(vs, 1), torch.stack(ws, 1)


def train_one(
    name: str,
    model: AccelMLP,
    flights: list[dict],
    pairs: list[tuple[int, int]],
    *,
    device: torch.device,
    x_mean: torch.Tensor,
    x_scale: torch.Tensor,
    scales: tuple[float, float, float, float],
) -> AccelMLP:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(
        f"{name:18s}  train {TRAIN_H} steps ({1000.0 * TRAIN_H / RATE_HZ:.0f} ms)  "
        f"eval {EVAL_H} steps (0.50 s)  stride {STRIDE}  "
        f"params {sum(p.numel() for p in model.parameters())}  "
        f"in {x_mean.numel()}  windows {len(pairs)}",
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
            f"  {name:18s}  epoch {epoch:02d}  "
            f"mean {mean.mean():.5f}  p {mean[0]:.5f}  R {mean[1]:.5f}  "
            f"v {mean[2]:.5f}  w {mean[3]:.5f}",
            flush=True,
        )
    model.eval()
    return model


@torch.no_grad()
def score_model(
    model: AccelMLP | None,
    flights: list[dict],
    *,
    device: torch.device,
    x_mean: torch.Tensor | None = None,
    x_scale: torch.Tensor | None = None,
    horizon: int = EVAL_H,
    stride: int = EVAL_H,
    coast: bool = False,
) -> dict[str, np.ndarray]:
    keys = ("position", "rotation", "velocity", "omega")
    totals = {k: np.zeros(horizon, dtype=np.float64) for k in keys}
    count = 0
    if model is not None:
        model.eval()
    for flight in flights:
        last = min(len(flight["dt"]), len(flight["omega"]) - 1) - horizon
        if last < 1:
            continue
        for start in range(0, last + 1, stride):
            p = torch.as_tensor(flight["position"][start : start + 1], device=device)
            r = torch.as_tensor(flight["rotation"][start : start + 1], device=device)
            v = torch.as_tensor(flight["vel_body"][start : start + 1], device=device)
            w = torch.as_tensor(flight["omega"][start : start + 1], device=device)
            motors = torch.as_tensor(flight["motors"][start : start + horizon], device=device)
            dt = torch.as_tensor(flight["dt"][start : start + horizon], device=device)
            for k in range(horizon):
                if not coast:
                    feat = (
                        _features_for(r, v, w, motors[k : k + 1]) - x_mean
                    ) / x_scale
                    acc = model(feat)
                    v = v + acc[:, 0:3] * dt[k]
                    w = w + acc[:, 3:6] * dt[k]
                p, r = _se3_exp_update(p, r, v, w, dt[k : k + 1])
                v_w = torch.bmm(r, v.unsqueeze(-1)).squeeze(-1)
                totals["position"][k] += float(
                    torch.linalg.vector_norm(
                        p[0] - torch.as_tensor(flight["position"][start + k + 1], device=device)
                    )
                )
                rel = torch.matmul(
                    torch.as_tensor(flight["rotation"][start + k + 1], device=device).T, r[0]
                )
                totals["rotation"][k] += float(torch.linalg.vector_norm(_so3_log(rel.unsqueeze(0))[0]))
                totals["velocity"][k] += float(
                    torch.linalg.vector_norm(
                        v_w[0] - torch.as_tensor(flight["vel_world"][start + k + 1], device=device)
                    )
                )
                totals["omega"][k] += float(
                    torch.linalg.vector_norm(
                        w[0] - torch.as_tensor(flight["omega"][start + k + 1], device=device)
                    )
                )
            count += 1
    n = max(count, 1)
    return {k: totals[k] / n for k in keys}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    train_paths, _val_paths, test_paths = _split(DEFAULT_ROOT)
    print("loading 400 Hz flights", flush=True)
    train = [_load(p) for p in train_paths]
    test = [_load(p) for p in test_paths]
    pairs = _windows(train, TRAIN_H, STRIDE)
    scales = _state_scales(train)
    print(f"train {len(train)}  test {len(test)}  windows {len(pairs)}  scales {scales}")

    # parity = Z2×Z2 four-orbit average (lr, fb, both); lr = left–right only
    specs = (
        ("quotient", None),
        ("quotient-lr", "lr"),
        ("quotient-parity", "parity"),
    )
    models: dict[str, tuple[AccelMLP, torch.Tensor, torch.Tensor]] = {}
    mean, scale = _feature_stats(train)
    mean, scale = mean.to(device), scale.to(device)
    for name, symmetry in specs:
        torch.manual_seed(0)
        model = AccelMLP(13, symmetry=symmetry)
        train_one(
            name,
            model,
            train,
            pairs,
            device=device,
            x_mean=mean,
            x_scale=scale,
            scales=scales,
        )
        torch.save(
            {
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "symmetry": symmetry,
                "x_mean": mean.cpu(),
                "x_scale": scale.cpu(),
            },
            OUT / f"fullstate_{name}.pt",
        )
        models[name] = (model, mean, scale)

    print("scoring 0.5 s official-test rollouts", flush=True)
    curves = {"coast": score_model(None, test, device=device, coast=True)}
    for name, (model, mean, scale) in models.items():
        curves[name] = score_model(
            model, test, device=device, x_mean=mean, x_scale=scale
        )

    t = np.arange(1, EVAL_H + 1) / RATE_HZ
    print(f"\n{'model':14s}  {'p h200':>8}  {'R h200':>8}  {'v h200':>8}  {'ω h200':>8}")
    for name, curve in curves.items():
        print(
            f"{name:14s}  {curve['position'][199]:8.4f}  {curve['rotation'][199]:8.4f}  "
            f"{curve['velocity'][199]:8.4f}  {curve['omega'][199]:8.4f}"
        )
        if name != "coast":
            better = np.where(curve["omega"] < curves["coast"]["omega"])[0]
            if len(better):
                print(f"  last ω beat of coast at t={t[better[-1]]:.3f}s")
            else:
                print("  ω never beats coast")

    np.savez(
        OUT / "fullstate_curves.npz",
        t=t,
        **{f"{name}_{key}": curve[key] for name, curve in curves.items() for key in curve},
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    style = {
        "coast": ("#222222", "--", 2.2, "coast (constant ν)"),
        "quotient": ("#c4504d", "-", 2.0, "quotient MLP"),
        "quotient-lr": ("#6d4c9e", "-", 2.0, "quotient + LR"),
        "quotient-parity": ("#2a9d8f", "-", 2.0, "quotient Z2×Z2"),
    }
    for ax, key, ylabel in (
        (axes[0], "omega", "ω MAE [rad/s]"),
        (axes[1], "position", "position MAE [m]"),
    ):
        for name, curve in curves.items():
            color, ls, lw, label = style[name]
            ax.plot(t, curve[key], color=color, ls=ls, lw=lw, label=label)
        ax.set_xlabel("horizon [s]")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Full SE(3) rollout: g ← g ∘ Exp(Δt ν), logged Ω only")
    fig.tight_layout()
    fig.savefig(OUT / "fullstate_vs_horizon.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'fullstate_vs_horizon.png'}")
    print(
        "\n---- compare ω at 0.5 s to teacher-forced ω-only MLP (logged v, not full SE(3)) ----"
    )
    print("  coast 1.501  plain-roll 0.912  sym-roll 0.891  (runs/neurobem_observability/)")
    for name in ("quotient", "quotient-lr", "quotient-parity"):
        if name in curves:
            print(f"  full SE(3) {name:16s}  ω h200 {curves[name]['omega'][199]:.4f}")


if __name__ == "__main__":
    main()
