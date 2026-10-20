"""Train the symmetric net plus an antisymmetric residual on 0.5 s open-loop."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from quadrotor.canonical import canonical_dim, output_dim
from quadrotor.evaluate import benchmark_mae
from quadrotor.neurobem_data import load_neurobem_prepared
from quadrotor.sequence import ResidualParityMotorNet
from quadrotor.train import train_open_loop


def _metrics(values: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        state: {
            "h10": float(curve[9]),
            "h50": float(curve[49]),
            "sum_h1_h50": float(curve[:50].sum()),
        }
        for state, curve in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--residual-width", type=int, default=8)
    parser.add_argument("--residual-weight-l2", type=float, default=0.1)
    parser.add_argument("--residual-output-l2", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/neurobem_parity_residual_rollout_20ep"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train, validation, test = load_neurobem_prepared(seed=args.seed)
    torch.manual_seed(args.seed)
    model = ResidualParityMotorNet(
        canonical_dim(include_voltage=True),
        output_dim(),
        hidden_per_type=24,
        residual_width=args.residual_width,
    ).to(device)
    model.reads_voltage = True
    model.residual_weight_l2 = args.residual_weight_l2
    model.residual_output_l2 = args.residual_output_l2
    residual_params = sum(parameter.numel() for parameter in model.residual.parameters())
    print(
        f"residual parity: {sum(p.numel() for p in model.parameters())} parameters "
        f"({residual_params} antisymmetric)  device {device}",
        flush=True,
    )

    started = time.perf_counter()
    result = train_open_loop(
        model,
        train,
        validation,
        test,
        epochs=args.epochs,
        lr=args.lr,
        rollout_every=5,
        batch_size=args.batch_size,
        stride=args.stride,
        horizon=50,
        horizon_weight=0.0,
        name="parity_res",
    )
    validation_mae = benchmark_mae(model, validation, horizon=50, batch_size=512)
    test_mae = benchmark_mae(model, test, horizon=50, batch_size=512)
    residual_rms = 0.0
    if model.last_residual is not None:
        residual_rms = float(model.last_residual.detach().square().mean().sqrt())
    records = {
        "config": {
            "dataset": "neurobem",
            "objective": "open-loop 0.5 s state MSE at 100 Hz",
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "stride": args.stride,
            "seed": args.seed,
            "device": str(device),
            "horizon_steps": 50,
            "horizon_seconds": 0.5,
            "residual_width": args.residual_width,
            "residual_weight_l2": args.residual_weight_l2,
            "residual_output_l2": args.residual_output_l2,
        },
        "models": {
            "parity_res": {
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "residual_parameters": residual_params,
                "residual_rms": residual_rms,
                "training_seconds": time.perf_counter() - started,
                "best_epoch": result.best_epoch,
                "best_validation_state_mse": result.best_error,
                "validation_mae": _metrics(validation_mae),
                "test_mae": _metrics(test_mae),
                "history": result.history,
            }
        },
    }
    torch.save(model.state_dict(), args.output / "parity_res.pt")
    (args.output / "results.json").write_text(json.dumps(records, indent=2))
    print(f"results written to {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
