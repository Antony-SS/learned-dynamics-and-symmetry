"""Train the one-pass parity-equivariant recurrent model on IDSIA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from quadrotor.canonical import canonical_dim, output_dim
from quadrotor.evaluate import benchmark_mae
from quadrotor.idsia_data import load_idsia_prepared
from quadrotor.sequence import ParityMotorNet
from quadrotor.train import train_open_loop


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
    parser.add_argument(
        "--hidden-per-type",
        type=int,
        default=32,
        help="hidden channels in each of the four parity classes",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/idsia_parity_gru_20ep")
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    train_flights, val_flights, test_flights = load_idsia_prepared()
    torch.manual_seed(args.seed)
    model = ParityMotorNet(
        canonical_dim(), output_dim(), hidden_per_type=args.hidden_per_type
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"parity model: {parameter_count} parameters, "
        f"{args.hidden_per_type} channels/type",
        flush=True,
    )

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
        name="parity",
    )
    validation = benchmark_mae(model, val_flights, horizon=50, batch_size=512)
    test = benchmark_mae(model, test_flights, horizon=50, batch_size=512)
    records = {
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
            "hidden_per_type": args.hidden_per_type,
        },
        "model": {
            "name": "one-pass-z2xz2-parity",
            "parameters": parameter_count,
            "training_seconds": time.perf_counter() - started,
            "best_epoch": result.best_epoch,
            "best_validation_state_mse": result.best_error,
            "history": result.history,
            "validation_mae": _metrics(validation),
            "test_mae": _metrics(test),
        },
    }
    torch.save(model.state_dict(), args.output / "parity.pt")
    (args.output / "results.json").write_text(json.dumps(records, indent=2))
    print(f"results written to {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
