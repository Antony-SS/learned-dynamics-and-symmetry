"""Train a 25k-parameter feedforward equivariant model on a quadrotor dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from quadrotor.canonical import canonical_dim, load_prepared, output_dim
from quadrotor.evaluate import benchmark_mae
from quadrotor.idsia_data import load_idsia_prepared
from quadrotor.neurobem_data import load_neurobem_prepared
from quadrotor.sequence import ParityMLP
from quadrotor.train import train_open_loop


class VoltageParityMLP(ParityMLP):
    reads_voltage = True

    def __init__(self) -> None:
        super().__init__(canonical_dim(include_voltage=True), output_dim())


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
    parser.add_argument(
        "--dataset",
        choices=("idsia", "nanobench-rich", "neurobem"),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.dataset == "idsia":
        train_flights, validation_flights, test_flights = load_idsia_prepared()
        model = ParityMLP(canonical_dim(), output_dim())
        default_stride = 1
        split = "Square+Random+Chirp runs 1-3 train, run 4 validation; Melon test"
    elif args.dataset == "nanobench-rich":
        train_flights, validation_flights, test_flights = load_prepared(
            data_rich=True, seed=args.seed
        )
        model = VoltageParityMLP()
        default_stride = 10
        split = "Category A plus flight-level trajectory-stratified Category B"
    else:
        train_flights, validation_flights, test_flights = load_neurobem_prepared(
            seed=args.seed
        )
        model = VoltageParityMLP()
        default_stride = 10
        split = "released testset.txt; remaining recordings split 80/20 by flight"

    stride = default_stride if args.stride is None else args.stride
    output = (
        Path(f"runs/{args.dataset}_symmetric_mlp_20ep")
        if args.output is None
        else args.output
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    result = train_open_loop(
        model,
        train_flights,
        validation_flights,
        test_flights,
        epochs=args.epochs,
        lr=args.lr,
        rollout_every=5,
        batch_size=args.batch_size,
        stride=stride,
        horizon=50,
        horizon_weight=0.0,
        name="sym-mlp",
    )
    validation = benchmark_mae(
        model, validation_flights, horizon=50, batch_size=512
    )
    test = benchmark_mae(model, test_flights, horizon=50, batch_size=512)
    records = {
        "config": {
            "dataset": args.dataset,
            "split": split,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "stride": stride,
            "seed": args.seed,
            "horizon_steps": 50,
            "horizon_seconds": 0.5,
            "recurrent": False,
            "width": 29,
            "depth": 2,
        },
        "model": {
            "name": "one-pass-z2xz2-parity-mlp",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": time.perf_counter() - started,
            "best_epoch": result.best_epoch,
            "best_validation_state_mse": result.best_error,
            "history": result.history,
            "validation_mae": _metrics(validation),
            "test_mae": _metrics(test),
        },
    }
    torch.save(model.state_dict(), output / "symmetric_mlp.pt")
    (output / "results.json").write_text(json.dumps(records, indent=2))
    print(f"results written to {output / 'results.json'}")


if __name__ == "__main__":
    main()
