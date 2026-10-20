"""Train and record the three coherent recurrent quadrotor models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from quadrotor.canonical import (
    absolute_dim,
    canonical_dim,
    load_prepared,
    output_dim,
)
from quadrotor.evaluate import benchmark_mae
from quadrotor.sequence import MotorNet
from quadrotor.train import train_open_loop


class AbsoluteModel(MotorNet):
    reads_pose = True
    reads_voltage = True

    def __init__(self) -> None:
        super().__init__(absolute_dim(include_voltage=True), output_dim())


class CanonicalModel(MotorNet):
    reads_voltage = True

    def __init__(self) -> None:
        super().__init__(canonical_dim(include_voltage=True), output_dim())


class SymmetricModel(MotorNet):
    reads_voltage = True

    def __init__(self) -> None:
        super().__init__(canonical_dim(include_voltage=True), output_dim(), mirror=True)


def _metrics(values: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    metrics = {}
    for state, curve in values.items():
        row = {
            "h1": float(curve[0]),
            "h10": float(curve[9]),
            "h50": float(curve[49]),
            f"h{len(curve)}": float(curve[-1]),
            f"sum_h1_h{len(curve)}": float(curve.sum()),
        }
        metrics[state] = row
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--data-rich", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/quadrotor_gru_20ep"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    train_flights, val_flights, test_flights = load_prepared(
        data_rich=args.data_rich, seed=args.seed
    )
    device = torch.device("cpu")
    records: dict[str, object] = {
        "config": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": str(device),
            "horizon_steps": args.horizon,
            "horizon_seconds": args.horizon * 0.01,
            "horizon_weight": 0.0,
            "stride": args.stride,
            "split": "data-rich-flight-level" if args.data_rich else "paper-A-only",
            "train_flights": len(train_flights),
            "validation_flights": len(val_flights),
            "train_windows": sum(
                max(0, (flight.n_steps - args.horizon) // args.stride + 1)
                for flight in train_flights
            ),
            "validation_windows": sum(
                max(0, (flight.n_steps - args.horizon) // args.stride + 1)
                for flight in val_flights
            ),
            "test_flights": len(test_flights),
        },
        "models": {},
    }

    constructors = {
        "absolute": AbsoluteModel,
        "canonical": CanonicalModel,
        "symmetric": SymmetricModel,
    }
    for name, constructor in constructors.items():
        torch.manual_seed(args.seed)
        model = constructor().to(device)
        started = time.perf_counter()
        result = train_open_loop(
            model,
            train_flights,
            val_flights,
            test_flights,
            epochs=args.epochs,
            lr=args.lr,
            rollout_every=5,
            batch_size=args.batch_size,
            stride=args.stride,
            horizon=args.horizon,
            horizon_weight=0.0,
            name=name,
        )
        training_seconds = time.perf_counter() - started
        validation = benchmark_mae(
            model, val_flights, horizon=args.horizon, batch_size=args.batch_size
        )
        test = benchmark_mae(
            model, test_flights, horizon=args.horizon, batch_size=args.batch_size
        )
        records["models"][name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": training_seconds,
            "best_epoch": result.best_epoch,
            "best_validation_state_mse": result.best_error,
            "history": result.history,
            "validation_mae": _metrics(validation),
            "test_mae": _metrics(test),
        }
        torch.save(model.state_dict(), args.output / f"{name}.pt")
        with (args.output / "results.json").open("w") as stream:
            json.dump(records, stream, indent=2)

    print(f"results written to {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
