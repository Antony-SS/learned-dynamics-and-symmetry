"""Regularizer and battery ablations on the 0.5 s NeuroBEM open-loop recipe."""

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
from quadrotor.sequence import ParityMotorNet, ResidualParityMotorNet
from quadrotor.train import _features, train_open_loop


def _metrics(values: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        state: {
            "h10": float(curve[9]),
            "h50": float(curve[49]),
            "sum_h1_h50": float(curve[:50].sum()),
        }
        for state, curve in values.items()
    }


def _residual_rms(model: torch.nn.Module, flights: list, samples: int = 128) -> float:
    if not hasattr(model, "antisymmetric_residual"):
        return 0.0
    device = next(model.parameters()).device
    values = []
    model.eval()
    with torch.inference_mode():
        for flight in flights:
            for start in range(0, max(1, flight.n_steps - 1), 40):
                voltage = None
                if getattr(model, "reads_voltage", False):
                    voltage = torch.as_tensor(
                        flight.voltage[start], device=device
                    ).view(1)
                features = _features(
                    model,
                    torch.as_tensor(flight.position[start], device=device).unsqueeze(0),
                    torch.as_tensor(flight.rotation[start], device=device).unsqueeze(0),
                    torch.as_tensor(flight.velocity_world[start], device=device).unsqueeze(0),
                    torch.as_tensor(flight.omega_body[start], device=device).unsqueeze(0),
                    torch.as_tensor(flight.pwm[start], device=device).unsqueeze(0),
                    voltage,
                )
                residual = model.antisymmetric_residual(features)
                values.append(residual.square().mean().sqrt())
                if len(values) >= samples:
                    return float(torch.stack(values).mean())
    return float(torch.stack(values).mean()) if values else 0.0


def _train_one(
    *,
    name: str,
    residual: bool,
    voltage: bool,
    weight_l2: float,
    output_l2: float,
    train,
    validation,
    test,
    epochs: int,
    batch_size: int,
    stride: int,
    seed: int,
    device: torch.device,
    output: Path,
) -> dict[str, object]:
    torch.manual_seed(seed)
    in_dim = canonical_dim(include_voltage=voltage)
    if residual:
        model = ResidualParityMotorNet(
            in_dim, output_dim(), hidden_per_type=24, residual_width=8
        )
        model.residual_weight_l2 = weight_l2
        model.residual_output_l2 = output_l2
        residual_params = sum(parameter.numel() for parameter in model.residual.parameters())
    else:
        model = ParityMotorNet(in_dim, output_dim(), hidden_per_type=24)
        residual_params = 0
    model.reads_voltage = voltage
    model = model.to(device)
    print(
        f"\n{name}: params {sum(p.numel() for p in model.parameters())}  "
        f"residual {residual_params}  voltage {voltage}  "
        f"wL2 {weight_l2:g}  oL2 {output_l2:g}",
        flush=True,
    )
    started = time.perf_counter()
    result = train_open_loop(
        model,
        train,
        validation,
        test,
        epochs=epochs,
        lr=1e-3,
        rollout_every=5,
        batch_size=batch_size,
        stride=stride,
        horizon=50,
        horizon_weight=0.0,
        name=name,
    )
    validation_mae = benchmark_mae(model, validation, horizon=50, batch_size=512)
    test_mae = benchmark_mae(model, test, horizon=50, batch_size=512)
    residual_rms = _residual_rms(model, validation)
    payload = {
        "residual": residual,
        "voltage": voltage,
        "residual_weight_l2": weight_l2,
        "residual_output_l2": output_l2,
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
    torch.save(model.state_dict(), output / f"{name}.pt")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/neurobem_parity_ablation"))
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated variant names; empty runs all",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train, validation, test = load_neurobem_prepared(seed=args.seed)
    variants = (
        ("res_free_volt", True, True, 0.0, 0.0),
        ("res_light_volt", True, True, 1e-3, 0.1),
        ("exact_novolt", False, False, 0.0, 0.0),
        ("res_free_novolt", True, False, 0.0, 0.0),
    )
    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    existing_path = args.output / "results.json"
    if existing_path.is_file():
        records = json.loads(existing_path.read_text())
        records.setdefault("models", {})
    else:
        records = {
            "config": {
                "objective": "open-loop 0.5 s state MSE at 100 Hz",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "stride": args.stride,
                "seed": args.seed,
                "device": str(device),
                "reference": {
                    "exact_volt": "runs/neurobem_gru24_20ep",
                    "res_heavy_volt": "runs/neurobem_parity_residual_rollout_20ep",
                },
            },
            "models": {},
        }
    for name, residual, voltage, weight_l2, output_l2 in variants:
        if wanted and name not in wanted:
            continue
        records["models"][name] = _train_one(
            name=name,
            residual=residual,
            voltage=voltage,
            weight_l2=weight_l2,
            output_l2=output_l2,
            train=train,
            validation=validation,
            test=test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            stride=args.stride,
            seed=args.seed,
            device=device,
            output=args.output,
        )
        (args.output / "results.json").write_text(json.dumps(records, indent=2))
    print(f"results written to {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
