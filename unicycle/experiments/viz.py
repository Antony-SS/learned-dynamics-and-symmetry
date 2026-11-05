"""Dataset plots and collection stills / gifs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from unicycle.experiments.data import Trajectory, commanded_vw, turn_counts
from unicycle.experiments.utils.params import VIZ


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
