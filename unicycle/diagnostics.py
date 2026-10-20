"""Plots of one-step error and of what the unicycle networks predict."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.collections import LineCollection

from unicycle.data import Trajectory, kept_steps
from unicycle.geometry import compose_relative, se2_exp, wheel_poses


def _landed_xy(start: np.ndarray, relative: np.ndarray) -> np.ndarray:
    cosine = np.cos(start[:, 2])
    sine = np.sin(start[:, 2])
    x = start[:, 0] + cosine * relative[:, 0] - sine * relative[:, 1]
    y = start[:, 1] + sine * relative[:, 0] + cosine * relative[:, 1]
    return np.stack((x, y), axis=1)


def _predict(
    model: torch.nn.Module,
    increments: np.ndarray,
    poses: np.ndarray | None = None,
) -> np.ndarray:
    features = np.asarray(increments, dtype=np.float32)
    if getattr(model, "uses_pose", False) and features.shape[-1] == 2:
        if poses is None:
            pose = np.zeros(features.shape[:-1] + (3,), dtype=np.float32)
        else:
            pose = np.asarray(poses, dtype=np.float32)
        features = np.concatenate((features, pose), axis=-1)
    features = torch.as_tensor(features, dtype=torch.float32)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(features).detach().cpu().numpy()
    finally:
        if was_training:
            model.train()


def step_errors(
    full_model: torch.nn.Module,
    residual_model: torch.nn.Module,
    equivariant_model: torch.nn.Module,
    absolute_model: torch.nn.Module,
    logs: list[Trajectory],
) -> list[dict]:
    """One-step world xy error (m) of each predictor, starting from the true pose."""
    records = []
    for log in logs:
        increments = np.asarray(log.increments, dtype=float)
        start = log.poses[:-1]
        target = log.poses[1:, :2]
        wheel = wheel_poses(increments)
        full = se2_exp(_predict(full_model, increments, start))
        residual = compose_relative(wheel, se2_exp(_predict(residual_model, increments, start)))
        equivariant = se2_exp(_predict(equivariant_model, increments, start))
        absolute = se2_exp(_predict(absolute_model, increments, start))

        def error(relative: np.ndarray) -> np.ndarray:
            landed = _landed_xy(start, relative)
            return np.hypot(landed[:, 0] - target[:, 0], landed[:, 1] - target[:, 1])

        records.append(
            {
                "name": log.name,
                "poses": log.poses,
                "v": increments[:, 0] / log.dt,
                "w": increments[:, 1] / log.dt,
                "keep": kept_steps(log),
                "dt": log.dt,
                "unicycle": error(wheel),
                "full": error(full),
                "residual": error(residual),
                "equivariant": error(equivariant),
                "absolute": error(absolute),
            }
        )
    return records


def plot_vw_errors(records: list[dict], plt) -> None:
    """Held-out steps in the (v, ω) plane, colored by one-step position error."""
    keep = np.concatenate([record["keep"] for record in records])
    v = np.concatenate([record["v"] for record in records])[keep]
    w = np.concatenate([record["w"] for record in records])[keep]
    fields = ("unicycle", "residual", "full", "equivariant", "absolute")
    titles = (
        "unicycle vs Vicon",
        "residual + unicycle vs Vicon",
        "full vs Vicon",
        "equivariant vs Vicon",
        "absolute pose vs Vicon",
    )
    stacked = [np.concatenate([record[name] for record in records])[keep] for name in fields]
    vmax = float(max(np.max(values) for values in stacked))
    fig, axes = plt.subplots(1, 5, figsize=(19.6, 3.8), constrained_layout=True)
    last = None
    for ax, values, title in zip(axes, stacked, titles):
        last = ax.scatter(
            v,
            w,
            c=values * 1e3,
            s=8,
            cmap="magma",
            vmin=0.0,
            vmax=vmax * 1e3,
            linewidths=0,
            alpha=0.75,
        )
        ax.set_title(f"{title}\nmean {values.mean() * 1e3:.2f} mm")
        ax.set_xlabel("v (m/s)")
        ax.set_ylabel("ω (rad/s)")
        ax.grid(alpha=0.25)
    fig.colorbar(last, ax=axes, label="one-step xy error (mm)")
    fig.suptitle("Held-out steps")
    plt.show()


def plot_error_trajectories(records: list[dict], plt) -> None:
    """Color the Vicon path by one-step error minus the unicycle's error.

    Negative means that step landed closer to Vicon than the wheel unicycle.
    """
    fields = ("residual", "full", "equivariant", "absolute")
    titles = (
        "residual − unicycle",
        "full − unicycle",
        "equivariant − unicycle",
        "absolute pose − unicycle",
    )
    deltas = [
        (record[name] - record["unicycle"])[record["keep"]]
        for record in records
        for name in fields
    ]
    limit = float(max(np.max(np.abs(delta)) for delta in deltas))
    limit = max(limit, 1e-9)
    norm = plt.Normalize(-limit * 1e3, limit * 1e3)
    rows = len(records)
    fig, axes = plt.subplots(rows, 4, figsize=(14.6, 2.7 * rows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, record in enumerate(records):
        poses = record["poses"]
        segments = np.stack((poses[:-1, :2], poses[1:, :2]), axis=1)
        keep = record["keep"]
        for col, (name, title) in enumerate(zip(fields, titles)):
            ax = axes[row, col]
            delta = (record[name] - record["unicycle"])[keep] * 1e3
            collection = LineCollection(
                segments[keep], cmap="coolwarm", norm=norm, linewidth=1.8, capstyle="round"
            )
            collection.set_array(delta)
            ax.add_collection(collection)
            ax.autoscale()
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.25)
            ax.set_xlabel("x (m)")
            heading = title if row == 0 else ""
            log_name = record["name"].replace("_", " ")
            ax.set_title(f"{heading}\n{log_name}" if heading else log_name, fontsize=9)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="coolwarm"),
        ax=axes,
        label="one-step xy error minus unicycle (mm)",
    )
    fig.suptitle("Held-out path. Blue is closer to Vicon than the unicycle, red is farther.")
    plt.show()


def _grid(logs: list[Trajectory], count: int = 81) -> tuple[np.ndarray, np.ndarray, float]:
    increments = np.concatenate(
        [log.increments[kept_steps(log)] for log in logs],
        axis=0,
    )
    dt = float(logs[0].dt)
    if any(log.dt != dt for log in logs):
        raise ValueError("every log must share the same time step")
    v = increments[:, 0] / dt
    w = increments[:, 1] / dt
    v_lo, v_hi = np.quantile(v, [0.01, 0.99])
    w_lo, w_hi = np.quantile(w, [0.01, 0.99])
    return (
        np.linspace(float(v_lo), float(v_hi), count),
        np.linspace(float(w_lo), float(w_hi), count),
        dt,
    )


def _held_command(model: torch.nn.Module, increments: np.ndarray) -> np.ndarray:
    """Predict one step for each wheel command on the probe grid."""
    return _predict(model, increments)


def _on_grid(model: torch.nn.Module, v_axis: np.ndarray, w_axis: np.ndarray, dt: float) -> np.ndarray:
    vv, ww = np.meshgrid(v_axis, w_axis, indexing="xy")
    increments = np.stack((vv.ravel() * dt, ww.ravel() * dt), axis=1)
    return _held_command(model, increments).reshape(vv.shape + (3,))


def plot_vw_heatmaps(
    full_model: torch.nn.Module,
    residual_model: torch.nn.Module,
    equivariant_model: torch.nn.Module,
    absolute_model: torch.nn.Module,
    logs: list[Trajectory],
    plt,
) -> None:
    """Learned body step over the (v, ω) range seen in the logs."""
    v_axis, w_axis, dt = _grid(logs)
    vv, ww = np.meshgrid(v_axis, w_axis, indexing="xy")
    extent = [v_axis[0], v_axis[-1], w_axis[0], w_axis[-1]]
    residual = _on_grid(residual_model, v_axis, w_axis, dt)
    full = _on_grid(full_model, v_axis, w_axis, dt)
    equivariant = _on_grid(equivariant_model, v_axis, w_axis, dt)
    absolute = _on_grid(absolute_model, v_axis, w_axis, dt)
    twist = np.stack((vv * dt, np.zeros_like(vv), ww * dt), axis=-1)
    full_correction = full - twist
    equivariant_correction = equivariant - twist
    absolute_correction = absolute - twist
    rows = (
        ("residual output", residual, True),
        ("full output", full, False),
        ("equivariant output", equivariant, False),
        ("absolute at origin", absolute, False),
        ("full correction", full_correction, True),
        ("equivariant correction", equivariant_correction, True),
        ("absolute correction", absolute_correction, True),
    )
    labels = ("body x (m)", "body y (m)", "yaw (rad)")
    fig, axes = plt.subplots(7, 3, figsize=(10.6, 18.4), constrained_layout=True)
    for row, (title, values, share) in enumerate(rows):
        for col in range(3):
            ax = axes[row, col]
            image = values[:, :, col]
            if share:
                limit = max(
                    float(np.max(np.abs(residual[:, :, col]))),
                    float(np.max(np.abs(full_correction[:, :, col]))),
                    float(np.max(np.abs(equivariant_correction[:, :, col]))),
                    float(np.max(np.abs(absolute_correction[:, :, col]))),
                    1e-12,
                )
            else:
                limit = max(
                    float(np.max(np.abs(full[:, :, col]))),
                    float(np.max(np.abs(equivariant[:, :, col]))),
                    float(np.max(np.abs(absolute[:, :, col]))),
                    1e-12,
                )
            picture = ax.imshow(
                image,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            fig.colorbar(picture, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel("v (m/s)")
            ax.set_ylabel("ω (rad/s)")
            ax.set_title(f"{title}: {labels[col]}", fontsize=9)
    fig.suptitle(
        "Network output on a v, ω grid. The absolute-pose rows are evaluated at (0, 0, 0). "
        "A correction is that output minus (v·dt, 0, ω·dt)."
    )
    plt.show()


def plot_output_direction(
    full_model: torch.nn.Module,
    residual_model: torch.nn.Module,
    equivariant_model: torch.nn.Module,
    absolute_model: torch.nn.Module,
    v: float,
    w: float,
    dt: float,
    plt,
) -> None:
    """Unit body-xy direction of each network output at one chosen (v, ω)."""
    increments = np.array([[v * dt, w * dt]], dtype=float)
    full = _held_command(full_model, increments)[0]
    residual = _held_command(residual_model, increments)[0]
    equivariant = _held_command(equivariant_model, increments)[0]
    absolute = _held_command(absolute_model, increments)[0]
    unicycle = np.array([v * dt, 0.0, w * dt])
    series = (
        ("unicycle", unicycle, "C1"),
        ("full", full, "C0"),
        ("residual", residual, "C2"),
        ("equivariant", equivariant, "C4"),
        ("absolute", absolute, "C3"),
    )
    print(f"query v = {v:g} m/s, ω = {w:g} rad/s, dt = {dt:g} s")
    print(f"{'output':12s} {'dx':>12s} {'dy':>12s} {'dyaw':>12s}")
    fig, ax = plt.subplots(figsize=(5.6, 5.4), constrained_layout=True)
    for name, vector, color in series:
        print(f"{name:12s} {vector[0]:12.4e} {vector[1]:12.4e} {vector[2]:12.4e}")
        if float(np.linalg.norm(vector)) < 1e-12:
            ax.scatter([0.0], [0.0], color=color, s=30, label=f"{name}: zero output")
            continue
        planar = float(np.hypot(vector[0], vector[1]))
        if planar < 1e-12:
            ax.scatter([0.0], [0.0], color=color, s=30, label=f"{name}: almost pure yaw")
            continue
        direction = vector[:2] / planar
        ax.quiver(
            0.0,
            0.0,
            direction[0],
            direction[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            width=0.014,
            label=f"{name}: δθ={vector[2]:+.3e} rad",
        )
    circle = plt.Circle((0.0, 0.0), 1.0, fill=False, linestyle=":", color="0.5")
    ax.add_patch(circle)
    ax.axhline(0.0, color="0.85", lw=0.8)
    ax.axvline(0.0, color="0.85", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel("body x direction")
    ax.set_ylabel("body y direction")
    ax.set_title(f"Unit (δx, δy) at v = {v:g} m/s, ω = {w:g} rad/s")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    plt.show()
