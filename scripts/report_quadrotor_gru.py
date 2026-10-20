"""Build a self-contained HTML report for a recurrent quadrotor experiment."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from quadrotor.canonical import (
    absolute_dim,
    canonical_dim,
    load_prepared,
    output_dim,
)
from quadrotor.evaluate import benchmark_mae
from quadrotor.rollout import open_loop
from quadrotor.sequence import MotorNet


class AbsoluteModel(MotorNet):
    reads_pose = True

    def __init__(self) -> None:
        super().__init__(absolute_dim(), output_dim())


class CanonicalModel(MotorNet):
    def __init__(self) -> None:
        super().__init__(canonical_dim(), output_dim())


class SymmetricModel(MotorNet):
    def __init__(self) -> None:
        super().__init__(canonical_dim(), output_dim(), mirror=True)


COLORS = {"absolute": "#b45309", "canonical": "#2563eb", "symmetric": "#7c3aed"}
UNITS = {
    "position": "m",
    "velocity": "m/s",
    "rotation": "rad",
    "omega": "rad/s",
}


def _image(fig: plt.Figure) -> str:
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(stream.getvalue()).decode()


def _load_models(run: Path) -> dict[str, MotorNet]:
    models = {
        "absolute": AbsoluteModel(),
        "canonical": CanonicalModel(),
        "symmetric": SymmetricModel(),
    }
    for name, model in models.items():
        model.load_state_dict(
            torch.load(run / f"{name}.pt", map_location="cpu", weights_only=True)
        )
        model.eval()
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/quadrotor_gru_20ep"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/quadrotor_gru_20ep.html")
    )
    args = parser.parse_args()
    records = json.loads((args.run / "results.json").read_text())
    horizon = int(records["config"]["horizon_steps"])
    _train, _val, test = load_prepared()
    models = _load_models(args.run)

    # Training curves.
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), sharey=True)
    for ax, (name, record) in zip(axes, records["models"].items()):
        epochs = [row["epoch"] for row in record["history"]]
        ax.plot(epochs, [row["train_loss"] for row in record["history"]],
                color=COLORS[name], label="train")
        ax.plot(epochs, [row["val_loss"] for row in record["history"]],
                color="#111827", linestyle="--", label="validation")
        ax.axvline(record["best_epoch"], color="#6b7280", linestyle=":", linewidth=1)
        ax.set_title(f"{name.capitalize()} · best epoch {record['best_epoch']}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Standardized full-state MSE")
    fig.suptitle("Training identifies early overfitting in every model", fontweight="bold")
    training_plot = _image(fig)

    # Exact Category B horizon curves and per-flight distributions.
    per_flight: dict[str, dict[str, list[np.ndarray]]] = {
        name: {state: [] for state in UNITS} for name in models
    }
    horizon_curves: dict[str, dict[str, np.ndarray]] = {
        name: {state: np.zeros(horizon) for state in UNITS} for name in models
    }
    total_windows = 0
    for flight in test:
        windows = flight.n_steps - horizon + 1
        if windows <= 0:
            continue
        total_windows += windows
        for name, model in models.items():
            metrics = benchmark_mae(model, [flight], horizon=horizon, batch_size=256)
            for state, curve in metrics.items():
                per_flight[name][state].append(curve)
                horizon_curves[name][state] += curve * windows
    for model_curves in horizon_curves.values():
        for state in model_curves:
            model_curves[state] /= total_windows

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), constrained_layout=True)
    for ax, state in zip(axes.ravel(), UNITS):
        for name in models:
            ax.plot(
                np.arange(1, horizon + 1) * 0.01,
                horizon_curves[name][state],
                color=COLORS[name],
                label=name,
            )
        ax.set_title(f"{state.capitalize()} vector-norm MAE")
        ax.set_xlabel("Open-loop time (s)")
        ax.set_ylabel(f"MAE ({UNITS[state]})")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Category B open-loop error over every valid window", fontweight="bold")
    horizon_plot = _image(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.5), constrained_layout=True)
    labels = list(models)
    for ax, state in zip(axes.ravel(), UNITS):
        values = [
            [curve[-1] for curve in per_flight[name][state]]
            for name in labels
        ]
        box = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, name in zip(box["boxes"], labels):
            patch.set_facecolor(COLORS[name])
            patch.set_alpha(0.6)
        ax.set_title(f"{state.capitalize()} at {horizon * 0.01:.1f} s")
        ax.set_ylabel(f"Per-flight MAE ({UNITS[state]})")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Category B variation across 103 held-out flights", fontweight="bold")
    distribution_plot = _image(fig)

    # Full-flight rollouts are a stress test, not the training benchmark.
    full_rmse = {name: [] for name in models}
    divergence = {name: [] for name in models}
    representative = sorted(range(len(test)), key=lambda i: test[i].n_steps)[len(test) // 2]
    representative_error: dict[str, np.ndarray] = {}
    for flight_index, flight in enumerate(test):
        for name, model in models.items():
            path, rmse = open_loop(model, flight)
            error = np.linalg.vector_norm(path - flight.position, axis=1)
            full_rmse[name].append(rmse)
            crossed = np.flatnonzero(error > 0.5)
            divergence[name].append(
                float(crossed[0] * 0.01) if crossed.size else flight.n_steps * 0.01
            )
            if flight_index == representative:
                representative_error[name] = error

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), constrained_layout=True)
    flight = test[representative]
    time = np.arange(flight.n_poses) * 0.01
    for name in models:
        axes[0].semilogy(
            time, np.maximum(representative_error[name], 1e-5),
            color=COLORS[name], label=name,
        )
    axes[0].set_title(f"Representative full flight: {flight.name}")
    axes[0].set_xlabel("Time from single Vicon seed (s)")
    axes[0].set_ylabel("Position error (m, log scale)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].boxplot(
        [divergence[name] for name in models],
        tick_labels=list(models),
        showfliers=False,
    )
    axes[1].set_title("Time until position error exceeds 0.5 m")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Divergence time (s)")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Full-flight stability stress test", fontweight="bold")
    full_plot = _image(fig)

    test_metrics = records["models"]
    symmetric_h10 = test_metrics["symmetric"]["test_mae"]["position"]["h10"]
    symmetric_h50 = test_metrics["symmetric"]["test_mae"]["position"]["h50"]
    paper_h10, paper_h50 = 0.0445, 0.2282
    improvement_h10 = 100.0 * (paper_h10 - symmetric_h10) / paper_h10
    improvement_h50 = 100.0 * (paper_h50 - symmetric_h50) / paper_h50
    rows = []
    for name, record in test_metrics.items():
        metric = record["test_mae"]
        rows.append(
            f"<tr><td>{name}</td><td>{record['best_epoch']}</td>"
            f"<td>{metric['position']['h10']:.4f}</td>"
            f"<td>{metric['position']['h50']:.4f}</td>"
            f"<td>{metric['rotation']['h50']:.4f}</td>"
            f"<td>{metric['omega']['h50']:.3f}</td></tr>"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quadrotor recurrent dynamics experiment</title>
<style>
body {{ margin:0; background:#f7f7f5; color:#171717; font:15px/1.55 system-ui,sans-serif; }}
main {{ max-width:1120px; margin:auto; padding:42px 24px 72px; }}
h1 {{ font-size:32px; margin:0 0 8px; }} h2 {{ margin-top:38px; }}
.subtle {{ color:#666; }} .hero {{ border-bottom:1px solid #d4d4d4; padding-bottom:28px; }}
.cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:24px 0; }}
.card {{ background:white; border:1px solid #ddd; border-radius:8px; padding:17px; }}
.value {{ font-size:27px; font-weight:700; color:#5b21b6; }}
.label {{ color:#666; font-size:13px; }}
.finding {{ border-left:4px solid #7c3aed; padding:3px 0 3px 16px; margin:22px 0; }}
img {{ width:100%; background:white; border:1px solid #ddd; border-radius:8px; }}
table {{ width:100%; border-collapse:collapse; background:white; }}
th,td {{ text-align:right; padding:10px; border-bottom:1px solid #e5e5e5; }}
th:first-child,td:first-child {{ text-align:left; }}
.note {{ background:#eeeae3; padding:14px 16px; border-radius:6px; }}
@media(max-width:760px) {{ .cards {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<section class="hero"><h1>Coherent recurrent quadrotor dynamics</h1>
<div class="subtle">20 epochs · 0.5 s uniform open-loop loss · NanoBench Category B held out · generated from {args.run}</div>
<div class="cards">
<div class="card"><div class="value">{symmetric_h10:.4f} m</div><div class="label">symmetric Category B position MAE at h=10</div></div>
<div class="card"><div class="value">{symmetric_h50:.4f} m</div><div class="label">symmetric Category B position MAE at h=50</div></div>
<div class="card"><div class="value">epoch 2</div><div class="label">best symmetric full-state validation checkpoint</div></div>
</div></section>
<div class="finding"><strong>Position benchmark:</strong> the symmetric model is {improvement_h10:.1f}% below the paper residual model at h=10 and {improvement_h50:.1f}% below it at h=50. This is promising, but not an exact reproduction because the released local training volume and unavailable preprocessing differ.</div>
<h2>Held-out Category B summary</h2>
<table><thead><tr><th>model</th><th>best epoch</th><th>position h10 (m)</th><th>position h50 (m)</th><th>rotation h50 (rad)</th><th>omega h50 (rad/s)</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Overfitting</h2><img alt="Training and validation curves" src="data:image/png;base64,{training_plot}">
<p>The absolute model memorizes fastest. Canonicalization materially lowers validation error; exact symmetry improves it further, but every model reaches its best checkpoint early.</p>
<h2>Error growth over the 0.5 s benchmark</h2><img alt="Category B horizon curves" src="data:image/png;base64,{horizon_plot}">
<h2>Variation between Category B flights</h2><img alt="Category B per-flight distributions" src="data:image/png;base64,{distribution_plot}">
<h2>Full-flight stress test</h2><img alt="Full-flight rollout stability" src="data:image/png;base64,{full_plot}">
<p class="note"><strong>Interpretation:</strong> full-flight rollout starts from one Vicon state and then uses only motor commands. It is intentionally much longer than the 0.5 s training horizon, so it diagnoses stability but is not directly comparable to the paper's h=1/10/50 benchmark.</p>
<h2>Interactive trajectory playback</h2>
<p><a href="quadrotor_5s_rollout.html">Open synchronized five-second real-versus-estimated rollouts with XY paths and x/y/z traces.</a></p>
<h2>Bottom line</h2>
<p>Coherent integration makes short-horizon position extremely accurate, and removing absolute pose improves validation. Symmetry gives the best validation objective and Category B position among the structured models. The main unresolved failure is angular generalization: canonical and symmetric angular-velocity errors transfer poorly to Category B despite strong validation-tail results. The 2 s experiment tests whether longer rollout supervision fixes that tradeoff.</p>
</main></body></html>"""
    )
    print(f"report written to {args.output}")


if __name__ == "__main__":
    main()
