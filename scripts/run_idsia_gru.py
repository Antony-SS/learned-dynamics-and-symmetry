"""Train the coherent models on the official IDSIA benchmark split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from quadrotor.canonical import absolute_dim, canonical_dim, output_dim
from quadrotor.evaluate import benchmark_mae
from quadrotor.idsia_data import load_idsia_prepared
from quadrotor.sequence import MotorNet
from quadrotor.train import train_open_loop


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


def _metrics(values: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        state: {
            "h1": float(curve[0]),
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
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/idsia_gru_20ep"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    train_flights, val_flights, test_flights = load_idsia_prepared()
    records: dict[str, object] = {
        "config": {
            "dataset": "idsia-robotics/nanodrone-sysid-benchmark",
            "split": "Square+Random+Chirp runs 1-3 train, run 4 validation; Melon test",
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "stride": args.stride,
            "seed": args.seed,
            "horizon_steps": 50,
            "horizon_seconds": 0.5,
            "horizon_weight": 0.0,
            "train_flights": len(train_flights),
            "validation_flights": len(val_flights),
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
        model = constructor()
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
            horizon=50,
            horizon_weight=0.0,
            name=name,
        )
        validation = benchmark_mae(model, val_flights, horizon=50, batch_size=512)
        test = benchmark_mae(model, test_flights, horizon=50, batch_size=512)
        records["models"][name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": time.perf_counter() - started,
            "best_epoch": result.best_epoch,
            "best_validation_state_mse": result.best_error,
            "history": result.history,
            "validation_mae": _metrics(validation),
            "test_mae": _metrics(test),
        }
        torch.save(model.state_dict(), args.output / f"{name}.pt")
        (args.output / "results.json").write_text(json.dumps(records, indent=2))
    print(f"results written to {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
