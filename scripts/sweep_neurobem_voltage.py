"""Sweep battery voltage at inference on a frozen NeuroBEM rollout model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from quadrotor.canonical import canonical_dim, output_dim
from quadrotor.evaluate import benchmark_mae
from quadrotor.neurobem_data import load_neurobem_prepared
from quadrotor.sequence import ParityMotorNet, ResidualParityMotorNet


def _summarize(curves: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        state: {"h10": float(curve[9]), "h50": float(curve[49])}
        for state, curve in curves.items()
    }


def _load_model(path: Path, residual: bool) -> torch.nn.Module:
    constructor = ResidualParityMotorNet if residual else ParityMotorNet
    extra = {"residual_width": 8} if residual else {}
    model = constructor(
        canonical_dim(include_voltage=True),
        output_dim(),
        hidden_per_type=24,
        **extra,
    )
    model.reads_voltage = True
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/neurobem_gru24_20ep/gru24.pt"),
    )
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-stride", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/neurobem_gru24_20ep/voltage_sweep.json"),
    )
    args = parser.parse_args()

    train, _, test = load_neurobem_prepared(seed=args.seed)
    voltage = np.concatenate([flight.voltage for flight in train])
    quantiles = {
        "min": float(voltage.min()),
        "p05": float(np.quantile(voltage, 0.05)),
        "p50": float(np.quantile(voltage, 0.50)),
        "p95": float(np.quantile(voltage, 0.95)),
        "max": float(voltage.max()),
        "mean": float(voltage.mean()),
    }
    model = _load_model(args.checkpoint, args.residual)
    settings: list[tuple[str, float | None]] = [("measured", None)]
    settings.extend((name, value) for name, value in quantiles.items())

    records: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "residual": args.residual,
        "start_stride": args.start_stride,
        "train_voltage": quantiles,
        "sweep": {},
    }
    print(
        f"voltage on train: min {quantiles['min']:.3f}  "
        f"p50 {quantiles['p50']:.3f}  max {quantiles['max']:.3f} V",
        flush=True,
    )
    for name, value in settings:
        curves = benchmark_mae(
            model,
            test,
            horizon=50,
            batch_size=512,
            start_stride=args.start_stride,
            voltage_value=value,
        )
        summary = _summarize(curves)
        records["sweep"][name] = {
            "voltage": value,
            **summary,
        }
        print(
            f"{name:12s}  V {('meas' if value is None else f'{value:.3f}'):>6s}  "
            f"h10 p {summary['position']['h10']:.4f}  w {summary['omega']['h10']:.3f}  "
            f"h50 p {summary['position']['h50']:.4f}  w {summary['omega']['h50']:.3f}",
            flush=True,
        )
        args.output.write_text(json.dumps(records, indent=2))
    print(f"results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
