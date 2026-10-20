"""Screen compact equivariant temporal models on matched rollout objectives."""

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
from quadrotor.sequence import ParityMotorNet, ParityTemporalMLP
from quadrotor.train import train_open_loop


DEFAULT_CANDIDATES = ("gru8", "gru16", "gru24", "gru32", "fir8", "ema4")


def _load_dataset(
    dataset: str, seed: int
) -> tuple[list, list, list, int, int, str]:
    if dataset == "idsia":
        train, validation, test = load_idsia_prepared()
        return train, validation, test, canonical_dim(), 3, "IDSIA official flight split"
    if dataset == "nanobench-rich":
        train, validation, test = load_prepared(data_rich=True, seed=seed)
        return (
            train,
            validation,
            test,
            canonical_dim(include_voltage=True),
            10,
            "NanoBench flight-level data-rich split",
        )
    train, validation, test = load_neurobem_prepared(seed=seed)
    return (
        train,
        validation,
        test,
        canonical_dim(include_voltage=True),
        10,
        "NeuroBEM released test split; remaining flights 80/20",
    )


def _make_model(name: str, in_dim: int) -> torch.nn.Module:
    if name.startswith("gru"):
        width = int(name.removeprefix("gru"))
        model = ParityMotorNet(in_dim, output_dim(), hidden_per_type=width)
    elif name.startswith("fir"):
        taps = int(name.removeprefix("fir"))
        model = ParityTemporalMLP(
            in_dim,
            output_dim(),
            memory="fir",
            memory_size=taps,
            width=24,
            depth=2,
        )
    elif name.startswith("ema"):
        scales = int(name.removeprefix("ema"))
        model = ParityTemporalMLP(
            in_dim,
            output_dim(),
            memory="ema",
            memory_size=scales,
            width=24,
            depth=2,
        )
    else:
        raise ValueError(f"unknown candidate {name!r}")
    model.reads_voltage = in_dim == canonical_dim(include_voltage=True)
    return model


def _metrics(values: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        state: {
            "h10": float(curve[9]),
            "h50": float(curve[49]),
            "sum_h1_h50": float(curve[:50].sum()),
        }
        for state, curve in values.items()
    }


def _latency_us(model: torch.nn.Module, in_dim: int, iterations: int = 1000) -> float:
    generator = torch.Generator().manual_seed(123)
    features = torch.randn(1, in_dim, generator=generator)
    hidden = model.initial_hidden(1, features)
    model.eval()
    with torch.no_grad():
        for _ in range(100):
            _, hidden = model.step(features, hidden)
        started = time.perf_counter()
        for _ in range(iterations):
            _, hidden = model.step(features, hidden)
    return 1e6 * (time.perf_counter() - started) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("idsia", "nanobench-rich", "neurobem"),
        default="idsia",
    )
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    train, validation, test, in_dim, default_stride, split = _load_dataset(
        args.dataset, args.seed
    )
    stride = default_stride if args.stride is None else args.stride
    output = args.output or Path(f"runs/{args.dataset}_temporal_arch_search")
    output.mkdir(parents=True, exist_ok=True)
    candidate_names = tuple(
        name.strip() for name in args.candidates.split(",") if name.strip()
    )
    records: dict[str, object] = {
        "config": {
            "dataset": args.dataset,
            "split": split,
            "selection_data": "validation only",
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "stride": stride,
            "seed": args.seed,
            "horizon_steps": 50,
            "candidates": candidate_names,
        },
        "models": {},
    }

    for name in candidate_names:
        torch.manual_seed(args.seed)
        model = _make_model(name, in_dim)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"\n{name}: {parameters:,} parameters", flush=True)
        started = time.perf_counter()
        result = train_open_loop(
            model,
            train,
            validation,
            test,
            epochs=args.epochs,
            lr=args.lr,
            rollout_every=max(1, args.epochs),
            batch_size=args.batch_size,
            stride=stride,
            horizon=50,
            horizon_weight=0.0,
            name=name,
        )
        validation_mae = benchmark_mae(
            model, validation, horizon=50, batch_size=512
        )
        test_mae = benchmark_mae(model, test, horizon=50, batch_size=512)
        details: dict[str, object] = {}
        if isinstance(model, ParityTemporalMLP) and model.memory == "ema":
            update = torch.sigmoid(model.memory_logits.detach()).cpu().numpy()
            details["ema_update_fraction"] = update.tolist()
            details["ema_time_constant_seconds_at_100hz"] = (
                -0.01 / np.log1p(-update)
            ).tolist()
        records["models"][name] = {
            "parameters": parameters,
            "training_seconds": time.perf_counter() - started,
            "inference_step_us_cpu": _latency_us(model, in_dim),
            "best_epoch": result.best_epoch,
            "best_validation_state_mse": result.best_error,
            "validation_mae": _metrics(validation_mae),
            "test_mae": _metrics(test_mae),
            "history": result.history,
            **details,
        }
        torch.save(model.state_dict(), output / f"{name}.pt")
        (output / "results.json").write_text(json.dumps(records, indent=2))

    print(f"results written to {output / 'results.json'}")


if __name__ == "__main__":
    main()
