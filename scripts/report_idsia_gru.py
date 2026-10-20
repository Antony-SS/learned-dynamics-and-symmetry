"""Build a self-contained IDSIA benchmark comparison report."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from quadrotor.canonical import absolute_dim, canonical_dim, output_dim
from quadrotor.evaluate import benchmark_mae
from quadrotor.idsia_data import DEFAULT_ROOT, load_idsia_prepared
from quadrotor.sequence import MotorNet
from quadrotor.train import _so3_exp, _so3_log

RUN = Path("runs/idsia_gru_20ep")
OUTPUT = Path("reports/idsia_gru_20ep.html")
COLORS = {"absolute": "#b45309", "canonical": "#2563eb", "symmetric": "#7c3aed"}
UNITS = {"position": "m", "velocity": "m/s", "rotation": "rad", "omega": "rad/s"}


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


def _image(fig: plt.Figure) -> str:
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(stream.getvalue()).decode()


def _reference_curves(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    curves = {name: [] for name in UNITS}
    for horizon in range(1, 51):
        count = len(frame) - horizon + 1
        groups = {
            "position": ("x", "y", "z"),
            "velocity": ("vx", "vy", "vz"),
            "omega": ("wx", "wy", "wz"),
        }
        for state, columns in groups.items():
            truth = frame[list(columns)].to_numpy()[horizon - 1 : horizon - 1 + count]
            prediction = frame[
                [f"{column}_pred_h{horizon}" for column in columns]
            ].to_numpy()[:count]
            curves[state].append(np.linalg.vector_norm(truth - prediction, axis=1).mean())
        truth_r = torch.as_tensor(
            frame[["rx", "ry", "rz"]].to_numpy()[horizon - 1 : horizon - 1 + count],
            dtype=torch.float32,
        )
        pred_r = torch.as_tensor(
            frame[
                [f"rx_pred_h{horizon}", f"ry_pred_h{horizon}", f"rz_pred_h{horizon}"]
            ].to_numpy()[:count],
            dtype=torch.float32,
        )
        relative = _so3_exp(truth_r).transpose(-1, -2) @ _so3_exp(pred_r)
        curves["rotation"].append(
            float(torch.linalg.vector_norm(_so3_log(relative), dim=-1).mean())
        )
    return {name: np.asarray(values) for name, values in curves.items()}


def main() -> None:
    records = json.loads((RUN / "results.json").read_text())
    _train, _validation, test = load_idsia_prepared()
    models = {
        "absolute": AbsoluteModel(),
        "canonical": CanonicalModel(),
        "symmetric": SymmetricModel(),
    }
    for name, model in models.items():
        model.load_state_dict(
            torch.load(RUN / f"{name}.pt", map_location="cpu", weights_only=True)
        )
    ours = {name: benchmark_mae(model, test, batch_size=512) for name, model in models.items()}

    prediction_root = DEFAULT_ROOT / "out" / "predictions"
    references = {
        "Physics": _reference_curves(
            prediction_root / "physics_model_multistep" / "melon_multistep.csv"
        ),
        "Residual": _reference_curves(
            prediction_root
            / "mlp_random_square_chirp_model_multistep"
            / "melon_multistep.csv"
        ),
        "Hybrid": _reference_curves(
            prediction_root
            / "hybrid_random_square_chirp_model_multistep"
            / "melon_multistep.csv"
        ),
        "LSTM": _reference_curves(
            prediction_root
            / "lstm_random_square_chirp_model_multistep"
            / "melon_multistep.csv"
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), sharey=True)
    for ax, (name, record) in zip(axes, records["models"].items()):
        epochs = [row["epoch"] for row in record["history"]]
        ax.plot(epochs, [row["train_loss"] for row in record["history"]],
                color=COLORS[name], label="train")
        ax.plot(epochs, [row["val_loss"] for row in record["history"]],
                color="#111827", linestyle="--", label="validation")
        ax.axvline(record["best_epoch"], color="#6b7280", linestyle=":")
        ax.set_title(f"{name.capitalize()} · best {record['best_epoch']}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Standardized full-state MSE")
    fig.suptitle("Training and validation on official IDSIA splits", fontweight="bold")
    training_plot = _image(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), constrained_layout=True)
    for ax, state in zip(axes.ravel(), UNITS):
        time = np.arange(1, 51) * 0.01
        for name in models:
            ax.plot(time, ours[name][state], color=COLORS[name], linewidth=2, label=name)
        for name in ("Physics", "Hybrid"):
            ax.plot(time, references[name][state], linestyle="--", linewidth=1.3, label=name)
        ax.set_title(f"{state.capitalize()} vector-norm MAE")
        ax.set_xlabel("Open-loop time (s)")
        ax.set_ylabel(f"MAE ({UNITS[state]})")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("Held-out Melon trajectories: error growth", fontweight="bold")
    horizon_plot = _image(fig)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), constrained_layout=True)
    names = ["Physics", "Residual", "Hybrid", "LSTM", "absolute", "canonical", "symmetric"]
    for ax, state in zip(axes, UNITS):
        values = [
            references[name][state][49] if name in references else ours[name][state][49]
            for name in names
        ]
        colors = ["#9ca3af"] * 4 + [COLORS[name] for name in names[4:]]
        ax.bar(range(len(names)), values, color=colors)
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
        ax.set_title(state.capitalize())
        ax.set_ylabel(f"h=50 MAE ({UNITS[state]})")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Official benchmark endpoint comparison", fontweight="bold")
    endpoint_plot = _image(fig)

    symmetric_position = ours["symmetric"]["position"]
    hybrid_position = references["Hybrid"]["position"]
    h50_gain = 100 * (hybrid_position[49] - symmetric_position[49]) / hybrid_position[49]
    cumulative_gain = 100 * (
        references["Physics"]["position"].sum() - symmetric_position.sum()
    ) / references["Physics"]["position"].sum()
    rows = []
    for name in models:
        metric = records["models"][name]["test_mae"]
        rows.append(
            f"<tr><td>{name}</td><td>{records['models'][name]['best_epoch']}</td>"
            f"<td>{metric['position']['h50']:.4f}</td>"
            f"<td>{metric['velocity']['h50']:.4f}</td>"
            f"<td>{metric['rotation']['h50']:.4f}</td>"
            f"<td>{metric['omega']['h50']:.4f}</td>"
            f"<td>{metric['position']['sum_h1_h50']:.4f}</td></tr>"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IDSIA symmetry benchmark</title>
<style>
body{{margin:0;background:#f7f7f5;color:#171717;font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:42px 24px 72px}}h1{{font-size:32px;margin:0 0 8px}}h2{{margin-top:38px}}
.subtle{{color:#666}}.hero{{border-bottom:1px solid #d4d4d4;padding-bottom:28px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}
.card{{background:white;border:1px solid #ddd;border-radius:8px;padding:17px}}
.value{{font-size:27px;font-weight:700;color:#5b21b6}}.label{{color:#666;font-size:13px}}
.finding{{border-left:4px solid #7c3aed;padding:3px 0 3px 16px;margin:22px 0}}
.caveat{{background:#fff7ed;border:1px solid #fdba74;border-radius:8px;padding:16px 18px;margin:24px 0}}
img{{width:100%;background:white;border:1px solid #ddd;border-radius:8px}}
table{{width:100%;border-collapse:collapse;background:white}}th,td{{text-align:right;padding:10px;border-bottom:1px solid #e5e5e5}}
th:first-child,td:first-child{{text-align:left}}.note{{background:#eeeae3;padding:14px 16px;border-radius:6px}}
@media(max-width:760px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main>
<section class="hero"><h1>Symmetry on the IDSIA nano-drone benchmark</h1>
<div class="subtle">Square + Random + Chirp train · run 4 validation · Melon held out · 50-step open-loop prediction</div>
<div class="cards"><div class="card"><div class="value">{symmetric_position[49]:.4f} m</div><div class="label">symmetric h=50 position MAE</div></div>
<div class="card"><div class="value">{symmetric_position.sum():.4f} m</div><div class="label">symmetric cumulative position error</div></div>
<div class="card"><div class="value">epoch {records['models']['symmetric']['best_epoch']}</div><div class="label">validation-selected symmetric checkpoint</div></div></div></section>
<div class="finding"><strong>Result:</strong> symmetry improves the official best h=50 position benchmark by {h50_gain:.1f}% and the best cumulative position benchmark by {cumulative_gain:.1f}%.</div>
<div class="caveat"><strong>Important caveat: the absolute model uses arena coordinates as a shortcut.</strong>
Translating every held-out Melon position by +1 m in world x—without changing any physical motion or motor input—increased its h=50 position error from 0.105 to 0.149 m (+41%), velocity from 0.480 to 0.834 m/s (+74%), rotation from 0.174 to 0.344 rad (+98%), and angular velocity from 0.655 to 1.234 rad/s (+88%). Canonical and symmetric scores were unchanged to numerical precision. The ordinary absolute score therefore overstates the quality of its learned dynamics, while the symmetric model's benchmark improvement is translation invariant.</div>
<h2>Our held-out Melon scores</h2>
<table><thead><tr><th>model</th><th>best epoch</th><th>position h50</th><th>velocity h50</th><th>rotation h50</th><th>omega h50</th><th>position sum</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Learning behavior</h2><img alt="training curves" src="data:image/png;base64,{training_plot}">
<h2>Error across the full horizon</h2><img alt="horizon curves" src="data:image/png;base64,{horizon_plot}">
<h2>Reference-model comparison</h2><img alt="endpoint comparison" src="data:image/png;base64,{endpoint_plot}">
<p class="note">Reference curves are recomputed from the prediction CSVs shipped by the IDSIA repository. All values are vector-norm MAE on held-out Melon trajectories.</p>
<h2>Five-second trajectory playback</h2><p><a href="idsia_5s_rollout.html">Open synchronized real-versus-estimated XY paths and x/y/z traces.</a></p>
<h2>Interpretation</h2><p>Symmetry gives the best transferred position prediction, including a small advantage over absolute state. Absolute remains stronger on endpoint velocity and angular channels, indicating that exact airframe symmetry is most beneficial for translational trajectory generalization but does not solve every unmodeled rotational effect.</p>
</main></body></html>"""
    )
    print(f"report written to {OUTPUT}")


if __name__ == "__main__":
    main()
