"""Shared MAE tables and IDSIA-style rollout figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from quadrotor.canonical import HORIZON_STEPS, PreparedFlight
from quadrotor.evaluate import benchmark_mae
from quadrotor.geometry import roll_pitch_yaw
from quadrotor.train import _integrate, _so3_log, module_device, prime_hidden

HORIZONS = (("h=1", 0), ("h=10", 9), ("h=50", 49))
NEUROBEM_HORIZONS = (("h=1", 0), ("h=10", 9), ("h=50", 49), ("h=100", 99), ("h=200", 199))
STATES = (
    ("position", "position", "m"),
    ("velocity", "velocity", "m/s"),
    ("rotation", "rotation", "rad"),
    ("omega", "angular rate", "rad/s"),
)
DEFAULT_COLORS = {
    "symmetric": "#6d4c9e",
    "physics": "#5c5c5c",
    "phys+res": "#c4504d",
    "tcn": "#2a6f97",
    "residual": "#b5651d",
    "vicon": "#222222",
}


def score_model(
    model: torch.nn.Module,
    flights: list[PreparedFlight],
    *,
    start_stride: int = 1,
    start_offset: int = 0,
    horizon: int = HORIZON_STEPS,
    horizons: tuple[tuple[str, int], ...] = HORIZONS,
) -> dict[str, dict[str, float]]:
    metrics = benchmark_mae(
        model,
        flights,
        horizon=horizon,
        start_stride=start_stride,
        start_offset=start_offset,
    )
    row: dict[str, dict[str, float]] = {}
    for state, _label, _unit in STATES:
        series = metrics[state]
        row[state] = {key: float(series[index]) for key, index in horizons}
    return row


def print_mae(
    rows: dict[str, dict[str, dict[str, float]]],
    labels: dict[str, str],
    horizons: tuple[tuple[str, int], ...] = HORIZONS,
) -> None:
    keys = [key for key, _index in horizons]
    for state, label, unit in STATES:
        print(f"\n{label} [{unit}]")
        print(f"{'model':22s}  " + "  ".join(f"{key:>8s}" for key in keys))
        for name, row in rows.items():
            vals = row[state]
            print(
                f"{labels.get(name, name):22s}  "
                + "  ".join(f"{vals[key]:8.4f}" for key in keys)
            )


def write_mae_image(
    path: str | Path,
    rows: dict[str, dict[str, dict[str, float]]],
    labels: dict[str, str],
    title: str,
    summary: str = "h=50",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["model"]
    for state, label, unit in STATES:
        header.append(f"{label} {summary} [{unit}]")
    cell = [header]
    for name, row in rows.items():
        cell.append(
            [labels.get(name, name)]
            + [f"{row[state][summary]:.4f}" for state, _label, _unit in STATES]
        )
    fig, ax = plt.subplots(figsize=(10.6, 0.7 + 0.45 * len(cell)))
    ax.axis("off")
    ax.set_title(title, pad=12)
    table = ax.table(cellText=cell, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)
    for col in range(len(header)):
        table[0, col].set_facecolor("#ece8f4")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_seed_image(
    path: str | Path,
    stats: dict[str, dict[str, tuple[float, float]]],
    labels: dict[str, str],
    title: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", "position h=50 [m]", "rotation h=50 [rad]", "rate h=50 [rad/s]"]
    cell = [header]
    for name, row in stats.items():
        cell.append(
            [labels.get(name, name)]
            + [
                f"{row[key][0]:.4f} ± {row[key][1]:.4f}"
                for key in ("position", "rotation", "omega")
            ]
        )
    fig, ax = plt.subplots(figsize=(10.2, 0.7 + 0.45 * len(cell)))
    ax.axis("off")
    ax.set_title(title, pad=12)
    table = ax.table(cellText=cell, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)
    for col in range(len(header)):
        table[0, col].set_facecolor("#ece8f4")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def window_rollout(
    model: torch.nn.Module,
    flight: PreparedFlight,
    start: int,
    horizon: int = HORIZON_STEPS,
    hidden: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    device = module_device(model)
    stop = start + horizon
    pwm = torch.as_tensor(flight.pwm[start:stop][None], device=device)
    voltage = torch.as_tensor(flight.voltage[start:stop][None], device=device)
    dt = torch.as_tensor(flight.dt[start:stop][None], device=device)
    model.eval()
    with torch.no_grad():
        if hidden is None:
            warm = int(getattr(model, "history", 0) or 0)
            if warm:
                hidden = prime_hidden(model, [flight], [(0, start)], warm)
        position_hat, rotation_hat, velocity_hat, omega_hat = _integrate(
            model,
            pwm,
            dt,
            torch.as_tensor(flight.position[start : start + 1], device=device),
            torch.as_tensor(flight.rotation[start : start + 1], device=device),
            torch.as_tensor(flight.velocity_world[start : start + 1], device=device),
            torch.as_tensor(flight.omega_body[start : start + 1], device=device),
            voltage,
            hidden=hidden,
        )
    return {
        "position": np.concatenate(
            [flight.position[start : start + 1], position_hat[0].cpu().numpy()], axis=0
        ),
        "velocity": np.concatenate(
            [flight.velocity_world[start : start + 1], velocity_hat[0].cpu().numpy()],
            axis=0,
        ),
        "rotation": np.concatenate(
            [flight.rotation[start : start + 1], rotation_hat[0].cpu().numpy()], axis=0
        ),
        "omega": np.concatenate(
            [flight.omega_body[start : start + 1], omega_hat[0].cpu().numpy()], axis=0
        ),
    }


def window_errors(
    predicted: dict[str, np.ndarray], flight: PreparedFlight, start: int, horizon: int
) -> dict[str, np.ndarray]:
    stop = start + horizon
    relative = np.einsum(
        "nji,njk->nik",
        flight.rotation[start + 1 : stop + 1],
        predicted["rotation"][1:],
    )
    return {
        "position": np.linalg.norm(
            predicted["position"][1:] - flight.position[start + 1 : stop + 1], axis=-1
        ),
        "velocity": np.linalg.norm(
            predicted["velocity"][1:] - flight.velocity_world[start + 1 : stop + 1],
            axis=-1,
        ),
        "rotation": np.linalg.norm(
            _so3_log(torch.as_tensor(relative)).numpy(), axis=-1
        ),
        "omega": np.linalg.norm(
            predicted["omega"][1:] - flight.omega_body[start + 1 : stop + 1], axis=-1
        ),
    }


def pick_windows(
    flights: list[PreparedFlight],
    *,
    count: int = 3,
    seed: int = 0,
    horizon: int = HORIZON_STEPS,
    start_offset: int = 0,
) -> list[tuple[PreparedFlight, int]]:
    rng = np.random.default_rng(seed)
    candidates = [
        (flight, start)
        for flight in flights
        for start in range(start_offset, flight.n_steps - horizon + 1, max(1, flight.n_steps // 8))
    ]
    if not candidates:
        raise ValueError("no flight is long enough for the requested windows")
    chosen = [candidates[i] for i in rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)]
    return chosen


def plot_3d(
    windows: list[dict],
    labels: dict[str, str],
    colors: dict[str, str],
    title: str,
):
    fig = plt.figure(figsize=(4.4 * len(windows), 4.4))
    for index, window in enumerate(windows):
        ax = fig.add_subplot(1, len(windows), index + 1, projection="3d")
        flight = window["flight"]
        ctx0, ctx1 = window["ctx0"], window["ctx1"]
        context = flight.position[ctx0 : ctx1 + 1]
        ax.plot(context[:, 0], context[:, 1], context[:, 2], color="#c8c8c8", lw=1.2, label="Vicon context")
        gt = window["gt"]["position"]
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="#222222", lw=2.4, label="Vicon 0.5 s")
        ax.scatter(*gt[0], color="#222222", s=28, depthshade=False)
        for name, pred in window["predicted"].items():
            xyz = pred["position"]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=colors[name], lw=1.8, label=labels[name])
        ax.set_title(flight.name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.view_init(elev=22, azim=-60)
        try:
            ax.set_box_aspect(np.ptp(context, axis=0) + 1e-6)
        except TypeError:
            pass
        if index == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def _series(values: np.ndarray, key: str) -> np.ndarray:
    return roll_pitch_yaw(values) if key == "rotation" else values
