"""One figure per question in the symmetry notebook.

Paths, loss curves, and a ``(v, ω)`` response. Drawn with Plotly.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from unicycle.data import BASE_LENGTH, METERS_PER_TICK, Trajectory, kept_steps
from unicycle.rollout import open_loop, residual_open_loop, wheel_open_loop

VICON = "#1c1c1c"
UNICYCLE = "#c47b2b"
COLORS = {
    "naive": "#2f6f9f",
    "canonical": "#2a9d8f",
    "symmetric": "#6d4c9e",
    "front_back": "#8a5a44",
    "reversible": "#8a5a44",
    "even_odd": "#2f6b4f",
    "symmetric_asym": "#2f6b4f",
    "residual": "#c4504d",
}
# Loss charts keep this epoch axis even when early stopping ends a run sooner.
LOSS_EPOCH_AXIS = 100

_FONT = dict(family="Inter, Source Sans 3, sans-serif", size=13, color="#1c1c1c")


def _short(name: str) -> str:
    name = re.sub(
        r" \((?:first \d+%|last \d+%|last quarter|first three quarters|first half|second half|test)\)",
        "",
        name,
    )
    name = name.replace("_", " ").replace("lawnmower ", "")
    if name == "figure8":
        return "figure 8"
    return name


def _show(fig: go.Figure) -> None:
    fig.show()


def _style(fig: go.Figure, title: str, *, top: int = 72) -> None:
    fig.update_layout(
        template="plotly_white",
        font=_FONT,
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12)),
        margin=dict(l=48, r=24, t=top, b=48),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#d9d9d9", mirror=True, title_standoff=8)
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor="#d9d9d9", mirror=True, title_standoff=8)


def _grid(n: int) -> tuple[int, int]:
    if n <= 3:
        return 1, n
    return int(np.ceil(n / 2)), 2


def _axis_name(kind: str, index: int) -> str:
    """Plotly subplot axis name. The first axis has no numeric suffix."""
    return kind if index == 1 else f"{kind}{index}"


def _equal_aspect(fig: go.Figure, n: int) -> None:
    for index in range(1, n + 1):
        fig.layout[_axis_name("yaxis", index)].scaleanchor = _axis_name("x", index)
        fig.layout[_axis_name("yaxis", index)].scaleratio = 1


def _predict(model: torch.nn.Module, increments: np.ndarray, poses: np.ndarray | None = None) -> np.ndarray:
    features = np.asarray(increments, dtype=np.float32)
    if getattr(model, "uses_pose", False):
        if poses is None:
            pose = np.zeros(features.shape[:-1] + (3,), dtype=np.float32)
        else:
            pose = np.asarray(poses, dtype=np.float32)
        features = np.concatenate((features, pose), axis=-1)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(torch.as_tensor(features)).detach().cpu().numpy()
    finally:
        if was_training:
            model.train()


def _counts_from_vw(v: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
    """Left/right tick increments that the wheel model reads as ``(v, ω)``."""
    v_dt = np.asarray(v, dtype=float) * dt
    w_dt = np.asarray(w, dtype=float) * dt
    left_arc = v_dt - 0.5 * w_dt * BASE_LENGTH
    right_arc = v_dt + 0.5 * w_dt * BASE_LENGTH
    return np.stack((left_arc, right_arc), axis=-1) / METERS_PER_TICK


def _path_figure(logs: list[Trajectory], title: str, traces) -> None:
    """One equal-aspect panel per log. ``traces(log)`` returns ``(panel title, scatter list)``."""
    rows, cols = _grid(len(logs))
    panel_titles = []
    drawn = []
    for log in logs:
        panel_title, scatters = traces(log)
        panel_titles.append(panel_title)
        drawn.append(scatters)
    while len(panel_titles) < rows * cols:
        panel_titles.append("")
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=panel_titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.16 if rows <= 2 else 0.09,
    )
    seen: set[str] = set()
    for index, scatters in enumerate(drawn):
        row, col = divmod(index, cols)
        for scatter in scatters:
            name = scatter.name
            scatter.showlegend = name not in seen
            scatter.legendgroup = name
            seen.add(name)
            fig.add_trace(scatter, row=row + 1, col=col + 1)
        axis = index + 1
        fig.layout[_axis_name("xaxis", axis)].title = "x (m)"
        fig.layout[_axis_name("yaxis", axis)].title = "y (m)"
    for index in range(len(logs), rows * cols):
        row, col = divmod(index, cols)
        fig.update_xaxes(visible=False, row=row + 1, col=col + 1)
        fig.update_yaxes(visible=False, row=row + 1, col=col + 1)
    fig.update_annotations(font=dict(size=13, color="#1c1c1c"))
    _equal_aspect(fig, len(logs))
    _style(fig, title, top=72)
    fig.update_layout(
        height=400 * rows + 48,
        width=460 * cols,
        title=dict(text=title, x=0, xanchor="left", xref="container", y=1, yanchor="top", yref="container"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22 if rows == 1 else -0.055,
            x=0,
            font=dict(size=12),
        ),
        margin=dict(t=64, b=100 if rows == 1 else 64, l=48, r=24),
    )
    _show(fig)


def _line(x, y, *, name: str, color: str, dash: str | None = None, width: float = 2.0) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line=dict(color=color, width=width, dash=dash),
        hovertemplate="%{x:.3f}, %{y:.3f}<extra>" + name + "</extra>",
    )


def _at_baseline(log: Trajectory, baseline: float) -> Trajectory:
    """Same log, with yaw rebuilt from the wheel arcs at ``baseline`` meters.

    Forward speed stays on the nominal radius and gear ratio. Only the wheel
    baseline in ``ω·dt = (right − left) / baseline`` changes.
    """
    if baseline <= 0:
        raise ValueError(f"wheel baseline must be positive, got {baseline}")
    if log.counts is None:
        raise ValueError(f"{log.name} has no raw encoder increments")
    arc = np.asarray(log.counts, dtype=float) * METERS_PER_TICK
    forward = 0.5 * (arc[:, 0] + arc[:, 1])
    yaw = (arc[:, 1] - arc[:, 0]) / baseline
    return replace(log, increments=np.stack((forward, yaw), axis=1))


def plot_unicycle_rollouts(
    logs: list[Trajectory],
    baseline: float | None = None,
    *,
    set_label: str,
) -> None:
    """Wheel unicycle on one split. ``baseline`` is meters. ``None`` keeps 105 mm."""
    shown = logs if baseline is None else [_at_baseline(log, baseline) for log in logs]
    baseline_m = BASE_LENGTH if baseline is None else baseline
    title = (
        f"{set_label} · wheel unicycle, baseline {baseline_m * 1000:.1f} mm, "
        "open loop from the first Vicon pose"
    )

    def traces(log: Trajectory):
        path, rmse = wheel_open_loop(log)
        panel = f"{_short(log.name)}<br>position RMSE {rmse:.2f} m"
        return panel, [
            _line(log.poses[:, 0], log.poses[:, 1], name="Vicon", color=VICON, width=2.2),
            _line(path[:, 0], path[:, 1], name="unicycle", color=UNICYCLE, dash="dash", width=1.6),
        ]

    _path_figure(shown, title, traces)


def plot_rollouts(
    model,
    logs: list[Trajectory],
    *,
    residual: bool,
    color: str,
    label: str,
    set_label: str,
) -> None:
    """Open-loop paths for one learned model on one split."""
    rollout = residual_open_loop if residual else open_loop

    def traces(log: Trajectory):
        wheel, wheel_rmse = wheel_open_loop(log)
        path, rmse = rollout(model, log)
        title = f"{_short(log.name)}<br>unicycle {wheel_rmse:.2f} m · {label} {rmse:.2f} m"
        return title, [
            _line(log.poses[:, 0], log.poses[:, 1], name="Vicon", color=VICON, width=2.2),
            _line(wheel[:, 0], wheel[:, 1], name="unicycle", color=UNICYCLE, dash="dash", width=1.4),
            _line(path[:, 0], path[:, 1], name=label, color=color, width=1.7),
        ]

    _path_figure(logs, f"{set_label} · {label}, open loop from the first Vicon pose", traces)


def plot_loss(result, *, color: str, label: str, others: list | None = None) -> None:
    """Train and held-out window loss, with the saved checkpoint marked."""
    fig = go.Figure()
    for other, other_color, other_label in others or []:
        epochs = [row["epoch"] for row in other.history]
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[row["val_loss"] for row in other.history],
                name=f"{other_label} validation",
                mode="lines",
                line=dict(color=other_color, width=1.4),
            )
        )
    epochs = [row["epoch"] for row in result.history]
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=[row["train_loss"] for row in result.history],
            name=f"{label} train",
            mode="lines",
            line=dict(color=color, width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=[row["val_loss"] for row in result.history],
            name=f"{label} validation",
            mode="lines",
            line=dict(color=color, width=2.2, dash="dash"),
        )
    )
    best = result.history[result.best_epoch]["val_loss"]
    fig.add_vline(x=result.best_epoch, line_width=1, line_color=color)
    fig.add_trace(
        go.Scatter(
            x=[result.best_epoch],
            y=[best],
            mode="markers+text",
            text=[f"best path, epoch {result.best_epoch}"],
            textposition="top right",
            marker=dict(color=color, size=8),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_yaxes(type="log", title="5 s position MSE", showgrid=True, gridcolor="#eeeeee")
    fig.update_xaxes(title="epoch", range=[0, LOSS_EPOCH_AXIS], showgrid=False)
    _style(fig, f"{label} loss")
    fig.update_layout(height=380, width=720)
    _show(fig)


def _symmetric_axis(values: np.ndarray, n: int = 71) -> np.ndarray:
    """Probe axis centered at 0, spanning the larger of the 1% and 99% tails."""
    lo, hi = np.quantile(np.asarray(values, dtype=float), [0.01, 0.99])
    limit = float(max(abs(lo), abs(hi)))
    if limit <= 0:
        raise ValueError("probe axis has no span")
    return np.linspace(-limit, limit, n)


def _probe_axes(logs: list[Trajectory]) -> tuple[np.ndarray, np.ndarray, float]:
    increments = np.concatenate([log.increments[kept_steps(log)] for log in logs], axis=0)
    dt = float(logs[0].dt)
    speed = increments / dt
    return _symmetric_axis(speed[:, 0]), _symmetric_axis(speed[:, 1]), dt


def _velocity_grid(model, v_axis: np.ndarray, w_axis: np.ndarray, dt: float) -> np.ndarray:
    """Network output, in velocity units, on an input ``(v, ω)`` grid."""
    vv, ww = np.meshgrid(v_axis, w_axis, indexing="xy")
    command = _counts_from_vw(vv.ravel(), ww.ravel(), dt)
    return _predict(model, command).reshape(vv.shape + (3,)) / dt


def _heat_row(title: str, panels: list[tuple[str, np.ndarray]], v_axis: np.ndarray, w_axis: np.ndarray) -> None:
    """One heatmap per panel, with the probe axes labeled as network inputs."""
    fig = make_subplots(
        rows=1,
        cols=len(panels),
        horizontal_spacing=0.22,
        subplot_titles=[name for name, _image in panels],
    )
    for channel, (name, image) in enumerate(panels):
        limit = max(float(np.max(np.abs(image))), 1e-6)
        fig.add_trace(
            go.Heatmap(
                x=v_axis,
                y=w_axis,
                z=image,
                zmin=-limit,
                zmax=limit,
                colorscale="RdBu_r",
                colorbar=dict(len=0.55, thickness=12, tickformat=".3g", nticks=5, xpad=20),
                hovertemplate="input v %{x:.3f}<br>input ω %{y:.3f}<br>" + name + " %{z:.4f}<extra></extra>",
            ),
            row=1,
            col=channel + 1,
        )
        fig.update_xaxes(title_text="Input v (m/s)", title_standoff=14, row=1, col=channel + 1)
        fig.update_yaxes(
            title_text="Input ω (rad/s)",
            title_standoff=12,
            tickmode="array",
            tickvals=[-1.5, -0.75, 0.75, 1.5],
            row=1,
            col=channel + 1,
        )
    for channel in range(len(panels)):
        domain = fig.layout[_axis_name("xaxis", channel + 1)].domain
        fig.data[channel].colorbar.x = domain[1] + 0.045
        fig.data[channel].colorbar.xanchor = "left"
    fig.update_annotations(yshift=28, font=dict(size=14, color="#1c1c1c"))
    _style(fig, title, top=88)
    width = 560 * len(panels)
    fig.update_layout(
        height=540,
        width=width,
        title=dict(text=title, x=0, xanchor="left", xref="container", y=1, yanchor="top", yref="container"),
        margin=dict(t=108, r=120, l=72, b=64),
    )
    _show(fig)


def plot_response(model, logs: list[Trajectory], *, residual: bool, at_origin: bool = False) -> None:
    """Body velocity on a ``(v, ω)`` grid.

    The network is fed the tick increments that the wheel model would read as
    that ``(v, ω)``. The naive model is evaluated at the origin. A residual
    model shows the correction it adds on top of the unicycle.
    """
    v_axis, w_axis, dt = _probe_axes(logs)
    field = _velocity_grid(model, v_axis, w_axis, dt)
    title = "Residual added to the unicycle" if residual else "Predicted body velocity"
    if at_origin:
        title += " at the origin"
    prefix = "residual body" if residual else "predicted body"
    labels = (f"{prefix} vₓ", f"{prefix} vᵧ", f"{prefix} ω")
    _heat_row(title, [(name, field[:, :, channel]) for channel, name in enumerate(labels)], v_axis, w_axis)


def plot_asymmetry(model, logs: list[Trajectory], *, label: str) -> None:
    """Where the learned map breaks a mirror across the longitudinal axis.

    The mirror of an input ``(v, ω)`` is ``(v, −ω)``. A symmetric model would
    give ``vᵧ(v, ω) = −vᵧ(v, −ω)`` and ``ω(v, ω) = −ω(v, −ω)``. These two
    panels are those sums, so a perfect mirror is a blank plot.
    """
    v_axis, w_axis, dt = _probe_axes(logs)
    here = _velocity_grid(model, v_axis, w_axis, dt)
    mirrored = _velocity_grid(model, v_axis, -w_axis, dt)
    lateral = here[:, :, 1] + mirrored[:, :, 1]
    yaw = here[:, :, 2] + mirrored[:, :, 2]
    _heat_row(
        f"{label} · leftover after a longitudinal mirror",
        [
            ("vᵧ(v, ω) + vᵧ(v, −ω)", lateral),
            ("ω(v, ω) + ω(v, −ω)", yaw),
        ],
        v_axis,
        w_axis,
    )


def plot_summary(regimes: list[tuple], validation_logs: list[Trajectory]) -> None:
    """Validation loss and the position RMSE used to choose checkpoints.

    ``regimes`` is ``(label, model, result, residual, color)`` in display order.
    The model bars are that checkpoint's validation position RMSE. The unicycle
    bar is the open-loop RMSE on ``validation_logs``. Path figures are separate.
    """
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14)
    for label, _model, result, _residual, color in regimes:
        epochs = [row["epoch"] for row in result.history]
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[row["val_loss"] for row in result.history],
                name=label,
                mode="lines+markers",
                marker=dict(
                    size=[9 if epoch == result.best_epoch else 0 for epoch in epochs],
                    color=color,
                ),
                line=dict(color=color, width=2),
            ),
            row=1,
            col=1,
        )
    names = ["unicycle"] + [item[0] for item in regimes]
    scores = [_unicycle_rmse(validation_logs)] + [item[2].best_error for item in regimes]
    colors = [UNICYCLE] + [item[4] for item in regimes]
    fig.add_trace(
        go.Bar(
            x=names,
            y=scores,
            marker_color=colors,
            text=[f"{score:.2f}" for score in scores],
            textposition="outside",
            showlegend=False,
            hovertemplate="%{x}<br>position RMSE %{y:.3f} m<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(type="log", title_text="validation 5 s MSE", row=1, col=1, showgrid=True, gridcolor="#eeeeee")
    fig.update_xaxes(
        title_text="epoch",
        range=[0, LOSS_EPOCH_AXIS],
        row=1,
        col=1,
        showgrid=False,
        zeroline=False,
        linecolor="#d9d9d9",
        mirror=True,
    )
    fig.update_yaxes(
        title_text="position RMSE (m)",
        range=[0, max(scores) * 1.28],
        row=1,
        col=2,
        showgrid=False,
        zeroline=False,
        linecolor="#d9d9d9",
        mirror=True,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#d9d9d9", mirror=True, row=1, col=2)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False, linecolor="#d9d9d9", mirror=True, row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        font=_FONT,
        title=dict(
            text="Validation loss and position RMSE",
            x=0,
            xanchor="left",
            xref="container",
            y=1,
            yanchor="top",
            yref="container",
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, font=dict(size=12)),
        margin=dict(l=64, r=40, t=64, b=72),
        height=460,
        width=960,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    _show(fig)


def plot_test_rmse(
    names: list[str],
    full: list[float],
    short: list[float],
    colors: list[str],
) -> None:
    """Pooled full and 5 s test position RMSE. One bar per model, meters."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14)
    fig.add_trace(
        go.Bar(
            x=names,
            y=full,
            marker_color=colors,
            text=[f"{score:.2f}" for score in full],
            textposition="outside",
            showlegend=False,
            hovertemplate="%{x}<br>position RMSE %{y:.3f} m<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=names,
            y=short,
            marker_color=colors,
            text=[f"{score:.3f}" for score in short],
            textposition="outside",
            showlegend=False,
            hovertemplate="%{x}<br>5.0 s position RMSE %{y:.3f} m<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="position RMSE (m)",
        range=[0, max(full) * 1.28],
        row=1,
        col=1,
        showgrid=False,
        zeroline=False,
        linecolor="#d9d9d9",
        mirror=True,
    )
    fig.update_yaxes(
        title_text="5.0 s position RMSE (m)",
        range=[0, max(short) * 1.28],
        row=1,
        col=2,
        showgrid=False,
        zeroline=False,
        linecolor="#d9d9d9",
        mirror=True,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#d9d9d9", mirror=True, tickangle=-30)
    fig.update_layout(
        template="plotly_white",
        font=_FONT,
        title=dict(
            text="Test · position RMSE",
            x=0,
            xanchor="left",
            xref="container",
            y=1,
            yanchor="top",
            yref="container",
        ),
        margin=dict(l=64, r=40, t=64, b=96),
        height=460,
        width=960,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    _show(fig)


def plot_summary_paths(regimes: list[tuple], logs: list[Trajectory], *, set_label: str) -> None:
    """One row per model. Columns are the logs in this split. Vicon is on every panel."""
    specs: list[tuple[str, str, object | None, bool]] = [("unicycle", UNICYCLE, None, False)]
    for label, model, _result, residual, color in regimes:
        specs.append((label, color, model, residual))
    n_rows = len(specs)
    n_cols = len(logs)
    titles: list[str] = []
    panels: list[list[go.Scatter]] = []
    for label, color, model, residual in specs:
        rollout = None if model is None else (residual_open_loop if residual else open_loop)
        for log in logs:
            if rollout is None:
                path, rmse = wheel_open_loop(log)
            else:
                path, rmse = rollout(model, log)
            titles.append(f"{_short(log.name)}<br>{rmse:.2f} m")
            panels.append(
                [
                    _line(log.poses[:, 0], log.poses[:, 1], name="Vicon", color=VICON, width=2.2),
                    _line(
                        path[:, 0],
                        path[:, 1],
                        name=label,
                        color=color,
                        dash="dash" if rollout is None else None,
                        width=1.7,
                    ),
                ]
            )
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.028,
        vertical_spacing=0.07,
    )
    seen: set[str] = set()
    for index, scatters in enumerate(panels):
        row, col = divmod(index, n_cols)
        for scatter in scatters:
            scatter.showlegend = scatter.name not in seen
            scatter.legendgroup = scatter.name
            seen.add(scatter.name)
            fig.add_trace(scatter, row=row + 1, col=col + 1)
        axis = index + 1
        fig.layout[_axis_name("xaxis", axis)].title = "x (m)" if row == n_rows - 1 else ""
        fig.layout[_axis_name("yaxis", axis)].title = ""
    _equal_aspect(fig, len(panels))
    fig.update_annotations(font=dict(size=12, color="#1c1c1c"))
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        linecolor="#d9d9d9",
        mirror=True,
        title_standoff=6,
        tickfont=dict(size=11),
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        linecolor="#d9d9d9",
        mirror=True,
        title_standoff=4,
        tickfont=dict(size=11),
    )
    for row in range(1, n_rows + 1):
        fig.update_yaxes(showticklabels=True, row=row, col=1)
    for col in range(1, n_cols + 1):
        fig.update_xaxes(showticklabels=True, row=n_rows, col=col)
    fig.update_layout(
        template="plotly_white",
        font=_FONT,
        title=dict(
            text=f"{set_label} · one row per model, open loop from the first Vicon pose",
            x=0,
            xanchor="left",
            xref="container",
            y=1,
            yanchor="top",
            yref="container",
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.045, x=0, font=dict(size=12)),
        margin=dict(l=168, r=24, t=96, b=72),
        height=310 * n_rows + 140,
        width=268 * n_cols + 160,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    for row, (label, color, _model, _residual) in enumerate(specs):
        yaxis = fig.layout[_axis_name("yaxis", row * n_cols + 1)]
        fig.add_annotation(
            text=label,
            x=0,
            y=0.5 * (yaxis.domain[0] + yaxis.domain[1]),
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="middle",
            xshift=-72,
            showarrow=False,
            font=dict(size=14, color=color),
        )
    _show(fig)


def _unicycle_rmse(logs: list[Trajectory]) -> float:
    total = 0.0
    weight = 0.0
    for log in logs:
        _path, rmse = wheel_open_loop(log)
        total += (rmse**2) * log.n_poses
        weight += log.n_poses
    return float(np.sqrt(total / weight))


def plot_data_efficiency(
    eval_logs: list[Trajectory],
    fractions: list[float],
    series: list[tuple],
    epochs: list[int],
) -> None:
    """Mean test position RMSE against the training prefix of each collection.

    Checkpoints are still chosen on the training-collection validation quarter.
    ``eval_logs`` is the score that gets plotted, the three held-out test
    collections. ``fractions`` and ``epochs`` line up. Each entry of
    ``series`` is ``(label, color, mean rmse per prefix, std rmse per prefix)``.
    The unicycle does not train, so it is a horizontal line at its RMSE on
    the same logs.
    """
    percents = [100.0 * fraction for fraction in fractions]
    order = sorted(range(len(percents)), key=lambda index: percents[index])
    xs = [percents[index] for index in order]
    epoch_labels = [epochs[index] for index in order]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[_unicycle_rmse(eval_logs)] * len(xs),
            name="unicycle",
            mode="lines",
            line=dict(color=UNICYCLE, width=2, dash="dash"),
            hovertemplate="unicycle<br>%{y:.3f} m<extra></extra>",
        )
    )
    for label, color, means, stds in series:
        ordered_mean = [means[index] for index in order]
        ordered_std = [stds[index] for index in order]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ordered_mean,
                name=label,
                mode="lines+markers",
                marker=dict(size=9, color=color),
                line=dict(color=color, width=2),
                error_y=dict(
                    type="data",
                    array=ordered_std,
                    visible=True,
                    color=color,
                    thickness=1.2,
                    width=5,
                ),
                customdata=[[epoch, std] for epoch, std in zip(epoch_labels, ordered_std)],
                hovertemplate=(
                    label
                    + "<br>%{x:.0f}% of each collection<br>%{customdata[0]} epochs"
                    + "<br>%{y:.3f} ± %{customdata[1]:.3f} m<extra></extra>"
                ),
            )
        )
    fig.update_xaxes(
        title_text="training prefix of each collection",
        tickmode="array",
        tickvals=xs,
        ticktext=[f"{percent:.0f}%<br>{epoch} epochs" for percent, epoch in zip(xs, epoch_labels)],
        showgrid=False,
        zeroline=False,
        linecolor="#d9d9d9",
        mirror=True,
    )
    fig.update_yaxes(
        title_text="mean test position RMSE (m)",
        showgrid=True,
        gridcolor="#eeeeee",
        zeroline=False,
        linecolor="#d9d9d9",
        mirror=True,
        rangemode="tozero",
    )
    fig.update_layout(
        template="plotly_white",
        font=_FONT,
        title=dict(
            text="Data efficiency · held-out test RMSE, mean ± std over 5 seeds",
            x=0,
            xanchor="left",
            xref="container",
            y=1,
            yanchor="top",
            yref="container",
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, font=dict(size=12)),
        margin=dict(l=64, r=32, t=64, b=88),
        height=480,
        width=760,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    _show(fig)
