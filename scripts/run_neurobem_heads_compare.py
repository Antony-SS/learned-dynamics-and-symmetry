"""Train a free symmetric+antisymmetric net and a plain canonical net."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from quadrotor.canonical import canonical_dim, output_dim
from quadrotor.evaluate import benchmark_mae
from quadrotor.neurobem_data import load_neurobem_prepared
from quadrotor.sequence import EvenOddSplitNet, MotorNet
from quadrotor.train import _features, train_open_loop


class CanonicalMotorNet(MotorNet):
    reads_voltage = True

    def __init__(self) -> None:
        super().__init__(canonical_dim(include_voltage=True), output_dim(), mirror=False)


def _metrics(values: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        state: {"h10": float(curve[9]), "h50": float(curve[49])}
        for state, curve in values.items()
    }


def _head_magnitudes(model: torch.nn.Module, flights: list, samples: int = 96) -> dict[str, float]:
    device = next(model.parameters()).device
    symmetric = []
    antisymmetric = []
    total = []
    model.eval()
    with torch.inference_mode():
        for flight in flights:
            for start in range(0, max(1, flight.n_steps - 1), 40):
                voltage = None
                if getattr(model, "reads_voltage", False):
                    voltage = torch.as_tensor(flight.voltage[start], device=device).view(1)
                features = _features(
                    model,
                    torch.as_tensor(flight.position[start], device=device).unsqueeze(0),
                    torch.as_tensor(flight.rotation[start], device=device).unsqueeze(0),
                    torch.as_tensor(flight.velocity_world[start], device=device).unsqueeze(0),
                    torch.as_tensor(flight.omega_body[start], device=device).unsqueeze(0),
                    torch.as_tensor(flight.pwm[start], device=device).unsqueeze(0),
                    voltage,
                )
                hidden = model.initial_hidden(1, features)
                acceleration, _ = model.step(features, hidden)
                if getattr(model, "last_symmetric", None) is not None:
                    symmetric.append(model.last_symmetric.detach().abs().mean())
                    antisymmetric.append(model.last_residual.detach().abs().mean())
                else:
                    symmetric.append(acceleration.detach().abs().mean())
                    antisymmetric.append(acceleration.new_zeros(()))
                total.append(acceleration.detach().abs().mean())
                if len(total) >= samples:
                    break
            if len(total) >= samples:
                break
    return {
        "symmetric_mean_abs": float(torch.stack(symmetric).mean()),
        "antisymmetric_mean_abs": float(torch.stack(antisymmetric).mean()),
        "output_mean_abs": float(torch.stack(total).mean()),
    }


def _plot(records: dict[str, object], output: Path) -> None:
    names = list(records["models"].keys())
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 10.5), sharex=True, constrained_layout=True)
    colors = {"heads_free": "#1d4ed8", "canonical": "#b45309"}
    labels = {
        "heads_free": "symmetric + free antisymmetric",
        "canonical": "plain canonical",
    }

    for name in names:
        history = records["models"][name]["history"]
        epochs = [row["epoch"] for row in history]
        color = colors.get(name, "#334155")
        axes[0].plot(epochs, [row["train_loss"] for row in history], color=color, linewidth=1.6, label=f"{labels[name]} train")
        axes[0].plot(
            epochs,
            [row["val_loss"] for row in history],
            color=color,
            linewidth=1.6,
            linestyle="--",
            label=f"{labels[name]} val",
        )
        axes[1].plot(
            epochs,
            [row["val_position_mae_h50"] for row in history],
            color=color,
            linewidth=1.8,
            label=f"{labels[name]} position",
        )
        axes[1].plot(
            epochs,
            [row["val_omega_mae_h50"] for row in history],
            color=color,
            linewidth=1.4,
            linestyle=":",
            label=f"{labels[name]} ω",
        )
        axes[2].plot(
            epochs,
            [row.get("symmetric_mean_abs", float("nan")) for row in history],
            color=color,
            linewidth=1.8,
            label=f"{labels[name]} symmetric |a|",
        )
        if name == "heads_free":
            axes[2].plot(
                epochs,
                [row.get("antisymmetric_mean_abs", float("nan")) for row in history],
                color="#0f766e",
                linewidth=1.8,
                linestyle="--",
                label="antisymmetric |a|",
            )

    axes[0].set_ylabel("weighted state MSE")
    axes[0].set_title("Training and validation loss")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("0.5 s validation MAE")
    axes[1].set_title("Open-loop position (m) and angular rate (rad/s)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.25)
    axes[2].set_ylabel("mean |acceleration|")
    axes[2].set_title("Head magnitudes on held-out canonical features")
    axes[2].set_xlabel("epoch")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.25)
    figure_path = output / "heads_compare.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    print(f"figure written to {figure_path}", flush=True)


def _train(
    *,
    name: str,
    model: torch.nn.Module,
    train,
    validation,
    test,
    epochs: int,
    batch_size: int,
    stride: int,
    seed: int,
    output: Path,
) -> dict[str, object]:
    torch.manual_seed(seed)
    started = time.perf_counter()
    result = train_open_loop(
        model,
        train,
        validation,
        test,
        epochs=epochs,
        lr=1e-3,
        rollout_every=1,
        batch_size=batch_size,
        stride=stride,
        horizon=50,
        horizon_weight=0.0,
        name=name,
        extra_metrics=_head_magnitudes,
    )
    test_mae = benchmark_mae(model, test, horizon=50, batch_size=512)
    payload = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": time.perf_counter() - started,
        "best_epoch": result.best_epoch,
        "best_validation_state_mse": result.best_error,
        "test_mae": _metrics(test_mae),
        "history": result.history,
    }
    torch.save(model.state_dict(), output / f"{name}.pt")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/neurobem_heads_compare"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train, validation, test = load_neurobem_prepared(seed=args.seed)
    records: dict[str, object] = {
        "config": {
            "objective": "0.5 s open-loop state MSE at 100 Hz",
            "antisymmetric_regularizer": 0.0,
            "split": "two orbit GRUs, hidden 47, ~9.2k params each",
            "epochs": args.epochs,
            "device": str(device),
        },
        "models": {},
    }

    heads = EvenOddSplitNet(
        canonical_dim(include_voltage=True), hidden_dim=47
    ).to(device)
    heads.reads_voltage = True
    even_params = sum(parameter.numel() for parameter in heads.even.parameters())
    odd_params = sum(parameter.numel() for parameter in heads.odd.parameters())
    print(
        f"even/odd split: even {even_params}  odd {odd_params}  "
        f"total {even_params + odd_params}",
        flush=True,
    )
    records["models"]["heads_free"] = _train(
        name="heads_free",
        model=heads,
        train=train,
        validation=validation,
        test=test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stride=args.stride,
        seed=args.seed,
        output=args.output,
    )
    (args.output / "results.json").write_text(json.dumps(records, indent=2))

    canonical = CanonicalMotorNet().to(device)
    records["models"]["canonical"] = _train(
        name="canonical",
        model=canonical,
        train=train,
        validation=validation,
        test=test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stride=args.stride,
        seed=args.seed,
        output=args.output,
    )
    (args.output / "results.json").write_text(json.dumps(records, indent=2))
    _plot(records, args.output)
    print(f"results written to {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
