"""Dataset plots and collection stills / gifs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from unicycle.experiments.data import Trajectory, commanded_vw, turn_counts
from unicycle.experiments.utils.params import ROLLOUT_VIZ_STARTS_S, ROLLOUT_VIZ_WINDOW_S, VIZ


def ensure_viz(root: Path | None = None) -> Path:
    dest = Path(root) if root is not None else VIZ
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "stills").mkdir(exist_ok=True)
    (dest / "gifs").mkdir(exist_ok=True)
    return dest


def plot_split_balance(train, val, test, dest: Path | None = None) -> Path:
    dest = ensure_viz(dest)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, name, logs in zip(axes, ("train", "val", "test"), (train, val, test)):
        counts = turn_counts(logs)
        ax.bar(["left", "right", "straight"], [counts["left"], counts["right"], counts["straight"]], color="#4C72B0")
        ax.set_title(f"{name}\n{counts['steps']} steps")
    fig.suptitle("Commanded yaw: left vs right")
    fig.tight_layout()
    path = dest / "turn_balance.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_command_heatmap(logs: list[Trajectory], dest: Path | None = None, title: str = "train") -> Path:
    dest = ensure_viz(dest)
    v, w = commanded_vw(logs)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].scatter(v, w, s=2, alpha=0.15, c="#4C72B0", linewidths=0)
    axes[0].set_xlabel("v_cmd (m/s)")
    axes[0].set_ylabel("ω_cmd (rad/s)")
    axes[0].set_title("scatter")
    axes[1].hist2d(v, w, bins=40, cmap="magma")
    axes[1].set_xlabel("v_cmd (m/s)")
    axes[1].set_ylabel("ω_cmd (rad/s)")
    axes[1].set_title("heatmap")
    fig.suptitle(f"Commanded (v, ω) · {title}")
    fig.tight_layout()
    path = dest / f"command_vw_{title}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_still(log: Trajectory, dest: Path, split: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot(log.poses[:, 0], log.poses[:, 1], color="#4C72B0", lw=1.2)
    ax.scatter(log.poses[0, 0], log.poses[0, 1], c="k", s=18, zorder=3)
    ax.set_aspect("equal")
    ax.set_title(f"{split}: {log.name}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    path = dest / f"{split}_{_slug(log.name)}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def write_gif(log: Trajectory, dest: Path, split: str, max_frames: int = 180) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    poses = log.poses
    stride = max(int(np.ceil(len(poses) / max_frames)), 1)
    idx = np.arange(0, len(poses), stride)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.set_aspect("equal")
    pad = 0.08
    ax.set_xlim(poses[:, 0].min() - pad, poses[:, 0].max() + pad)
    ax.set_ylim(poses[:, 1].min() - pad, poses[:, 1].max() + pad)
    (trail,) = ax.plot([], [], color="#4C72B0", lw=1.4)
    (dot,) = ax.plot([], [], "ko", ms=4)
    ax.set_title(f"{split}: {log.name}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    def update(frame: int):
        k = int(idx[frame])
        trail.set_data(poses[: k + 1, 0], poses[: k + 1, 1])
        dot.set_data([poses[k, 0]], [poses[k, 1]])
        return trail, dot

    anim = FuncAnimation(fig, update, frames=len(idx), interval=40, blit=True)
    path = dest / f"{split}_{_slug(log.name)}.gif"
    anim.save(path, writer=PillowWriter(fps=20))
    plt.close(fig)
    return path


def _device(model) -> torch.device:
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


def _rollout_xy(
    model,
    log: Trajectory,
    *,
    start_s: float,
    window_s: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth and predicted XY for one open-loop segment."""
    start = min(max(int(round(start_s / log.dt)), 0), log.n_steps - 1)
    steps = min(max(int(round(window_s / log.dt)), 1), log.n_steps - start)
    stop = start + steps
    pose = torch.tensor(log.poses[start], dtype=torch.float32, device=device).unsqueeze(0)
    rates = torch.tensor(log.rates[start], dtype=torch.float32, device=device).unsqueeze(0)
    command = torch.tensor(log.commands[start:stop], dtype=torch.float32, device=device).unsqueeze(0)
    pred = model.rollout(pose, rates, command, log.dt).squeeze(0).cpu().numpy()
    gt = log.poses[start : stop + 1]
    pred_path = np.vstack((gt[0], pred))
    return gt, pred_path


@torch.no_grad()
def plot_test_rollouts(
    model,
    logs: list[Trajectory],
    family: str,
    dest: Path | None = None,
    *,
    window_s: float = ROLLOUT_VIZ_WINDOW_S,
    starts_s: tuple[float, ...] = ROLLOUT_VIZ_STARTS_S,
) -> Path:
    """30 s open-loop segments at several start times on each test log."""
    dest = ensure_viz(dest) / "rollouts"
    dest.mkdir(parents=True, exist_ok=True)
    device = _device(model)
    model.eval()
    n_logs = max(len(logs), 1)
    n_starts = len(starts_s)
    fig, axes = plt.subplots(n_logs, n_starts, figsize=(4.2 * n_starts, 4.2 * n_logs), squeeze=False)
    for row, log in enumerate(logs):
        for col, start_s in enumerate(starts_s):
            ax = axes[row, col]
            gt, pred = _rollout_xy(model, log, start_s=start_s, window_s=window_s, device=device)
            ax.plot(gt[:, 0], gt[:, 1], color="#9aa4b2", lw=2.0, label="ground truth")
            ax.plot(pred[:, 0], pred[:, 1], color="#c0392b", lw=1.4, ls="--", label="rollout")
            ax.scatter(gt[0, 0], gt[0, 1], c="k", s=18, zorder=3)
            ax.set_aspect("equal")
            ax.set_title(f"{log.name} · t={start_s:g}s")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
    axes[0, 0].legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle(
        f"{family} · {window_s:g} s rollouts at t = {', '.join(f'{s:g}' for s in starts_s)} s",
        fontsize=14,
        color="#1a365d",
    )
    fig.tight_layout()
    path = dest / f"{family}_test.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def write_collections(train, val, test, dest: Path | None = None) -> Path:
    dest = ensure_viz(dest)
    stills = dest / "stills"
    gifs = dest / "gifs"
    for split, logs in (("train", train), ("val", val), ("test", test)):
        for log in logs:
            plot_still(log, stills, split)
            write_gif(log, gifs, split)
    return dest


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)
