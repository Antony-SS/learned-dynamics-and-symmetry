"""Train compact coherent models with the NYU history-20/unroll-10 objective."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_neurobem_nyu_protocol import evaluate
from quadrotor.canonical import canonical_dim, output_dim
from quadrotor.neurobem_data import load_neurobem_prepared
from quadrotor.sequence import MotorNet, ParityMotorNet
from quadrotor.train import (
    _apply,
    _configure_input_normalization,
    _features,
    _so3_log,
    _state_scales,
)


class CanonicalModel(MotorNet):
    reads_voltage = True

    def __init__(self) -> None:
        super().__init__(canonical_dim(include_voltage=True), output_dim())


class SymmetricModel(MotorNet):
    reads_voltage = True

    def __init__(self) -> None:
        super().__init__(
            canonical_dim(include_voltage=True), output_dim(), mirror=True
        )


class ParityModel(ParityMotorNet):
    reads_voltage = True

    def __init__(self) -> None:
        super().__init__(canonical_dim(include_voltage=True), output_dim())


def _starts(
    flights: list, history: int, unroll: int, stride: int
) -> list[tuple[int, int]]:
    return [
        (flight_index, origin)
        for flight_index, flight in enumerate(flights)
        for origin in range(history - 1, flight.n_steps - unroll + 1, stride)
    ]


def _batch_loss(
    model: torch.nn.Module,
    flights: list,
    pairs: list[tuple[int, int]],
    *,
    history: int,
    unroll: int,
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    past_position, past_rotation, past_velocity, past_omega = [], [], [], []
    past_pwm, past_voltage = [], []
    position, rotation, velocity, omega = [], [], [], []
    pwm, voltage, dt = [], [], []
    rotation_target, velocity_target, omega_target = [], [], []
    for flight_index, origin in pairs:
        flight = flights[flight_index]
        history_start = origin - history + 1
        stop = origin + unroll
        past_position.append(flight.position[history_start:origin])
        past_rotation.append(flight.rotation[history_start:origin])
        past_velocity.append(flight.velocity_world[history_start:origin])
        past_omega.append(flight.omega_body[history_start:origin])
        past_pwm.append(flight.pwm[history_start:origin])
        past_voltage.append(flight.voltage[history_start:origin])
        position.append(flight.position[origin])
        rotation.append(flight.rotation[origin])
        velocity.append(flight.velocity_world[origin])
        omega.append(flight.omega_body[origin])
        pwm.append(flight.pwm[origin:stop])
        voltage.append(flight.voltage[origin:stop])
        dt.append(flight.dt[origin:stop])
        rotation_target.append(flight.rotation[origin + 1 : stop + 1])
        velocity_target.append(flight.velocity_world[origin + 1 : stop + 1])
        omega_target.append(flight.omega_body[origin + 1 : stop + 1])

    device = next(model.parameters()).device

    def tensor(values: list) -> torch.Tensor:
        return torch.as_tensor(np.stack(values), device=device)

    past_position_t = tensor(past_position)
    past_rotation_t = tensor(past_rotation)
    past_velocity_t = tensor(past_velocity)
    past_omega_t = tensor(past_omega)
    past_pwm_t = tensor(past_pwm)
    past_voltage_t = tensor(past_voltage)
    position_t = tensor(position)
    rotation_t = tensor(rotation)
    velocity_t = tensor(velocity)
    omega_t = tensor(omega)
    pwm_t = tensor(pwm)
    voltage_t = tensor(voltage)
    dt_t = tensor(dt)
    rotation_target_t = tensor(rotation_target)
    velocity_target_t = tensor(velocity_target)
    omega_target_t = tensor(omega_target)

    hidden = model.initial_hidden(len(pairs), position_t)
    # The current sample is consumed by the first prediction below, yielding
    # exactly 20 observed samples including the current one.
    for step in range(history - 1):
        features = _features(
            model,
            past_position_t[:, step],
            past_rotation_t[:, step],
            past_velocity_t[:, step],
            past_omega_t[:, step],
            past_pwm_t[:, step],
            past_voltage_t[:, step],
        )
        _, hidden = model.step(features, hidden)

    rotation_predictions, velocity_predictions, omega_predictions = [], [], []
    for step in range(unroll):
        features = _features(
            model,
            position_t,
            rotation_t,
            velocity_t,
            omega_t,
            pwm_t[:, step],
            voltage_t[:, step],
        )
        acceleration, hidden = model.step(features, hidden)
        position_t, rotation_t, velocity_t, omega_t = _apply(
            position_t,
            rotation_t,
            velocity_t,
            omega_t,
            acceleration,
            dt_t[:, step],
        )
        rotation_predictions.append(rotation_t)
        velocity_predictions.append(velocity_t)
        omega_predictions.append(omega_t)

    rotation_hat = torch.stack(rotation_predictions, dim=1)
    velocity_hat = torch.stack(velocity_predictions, dim=1)
    omega_hat = torch.stack(omega_predictions, dim=1)
    relative = torch.matmul(rotation_target_t.transpose(-1, -2), rotation_hat)
    errors = (
        _so3_log(relative) / scales[1],
        (velocity_hat - velocity_target_t) / scales[2],
        (omega_hat - omega_target_t) / scales[3],
    )
    return torch.cat([error.square() for error in errors], dim=-1).mean()


def _mean_loss(
    model: torch.nn.Module,
    flights: list,
    pairs: list[tuple[int, int]],
    *,
    history: int,
    unroll: int,
    batch_size: int,
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    total = torch.zeros(())
    seen = 0
    for start in range(0, len(pairs), batch_size):
        chosen = pairs[start : start + batch_size]
        loss = _batch_loss(
            model,
            flights,
            chosen,
            history=history,
            unroll=unroll,
            scales=scales,
        )
        total = total + loss.detach() * len(chosen)
        seen += len(chosen)
    return total / seen


def _train(
    model: torch.nn.Module,
    train_flights: list,
    validation_flights: list,
    *,
    epochs: int,
    history: int,
    unroll: int,
    stride: int,
    batch_size: int,
    learning_rate: float,
    name: str,
) -> tuple[list[dict[str, float]], int, float]:
    _configure_input_normalization(model, train_flights)
    train_pairs = _starts(train_flights, history, unroll, stride)
    validation_pairs = _starts(validation_flights, history, unroll, stride)
    print(
        f"{name:10s} history {history} unroll {unroll} stride {stride} "
        f"train {len(train_pairs)} val {len(validation_pairs)}",
        flush=True,
    )
    device = next(model.parameters()).device
    scales = _state_scales(train_flights, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    order = torch.arange(len(train_pairs))
    history_rows: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    for epoch in range(epochs + 1):
        started = time.perf_counter()
        if epoch == 0:
            model.eval()
            with torch.no_grad():
                train_loss = _mean_loss(
                    model,
                    train_flights,
                    train_pairs,
                    history=history,
                    unroll=unroll,
                    batch_size=batch_size,
                    scales=scales,
                )
        else:
            model.train()
            order = order[torch.randperm(order.numel())]
            total = torch.zeros(())
            seen = 0
            for start in range(0, order.numel(), batch_size):
                chosen = [
                    train_pairs[int(index)]
                    for index in order[start : start + batch_size]
                ]
                loss = _batch_loss(
                    model,
                    train_flights,
                    chosen,
                    history=history,
                    unroll=unroll,
                    scales=scales,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total = total + loss.detach() * len(chosen)
                seen += len(chosen)
            train_loss = total / seen
            scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_loss = _mean_loss(
                model,
                validation_flights,
                validation_pairs,
                history=history,
                unroll=unroll,
                batch_size=batch_size,
                scales=scales,
            )
        saved = float(validation_loss) < best_loss
        if saved:
            best_loss = float(validation_loss)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
            "seconds": time.perf_counter() - started,
        }
        history_rows.append(row)
        print(
            f"{name:10s} epoch {epoch:03d} train {float(train_loss):.6e} "
            f"val {float(validation_loss):.6e}"
            f"{' saved' if saved else ''}",
            flush=True,
        )
    model.load_state_dict(best_state)
    return history_rows, best_epoch, best_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--history", type=int, default=20)
    parser.add_argument("--unroll", type=int, default=10)
    parser.add_argument("--eval-horizon", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/neurobem_nyu_objective_20ep")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    train_flights, validation_flights, test_flights = load_neurobem_prepared(
        seed=args.seed
    )
    constructors = {
        "canonical": CanonicalModel,
        "symmetric": SymmetricModel,
        "parity": ParityModel,
    }
    results: dict[str, object] = {
        "config": vars(args) | {"output": str(args.output)},
        "split": "released 13-segment test set; remaining recordings 80/20 by flight",
        "models": {},
    }
    for name, constructor in constructors.items():
        torch.manual_seed(args.seed)
        model = constructor()
        started = time.perf_counter()
        rows, best_epoch, best_loss = _train(
            model,
            train_flights,
            validation_flights,
            epochs=args.epochs,
            history=args.history,
            unroll=args.unroll,
            stride=args.stride,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            name=name,
        )
        scores = evaluate(
            model,
            test_flights,
            history=args.history,
            horizon=args.eval_horizon,
            batch_size=args.batch_size,
        )
        results["models"][name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "training_seconds": time.perf_counter() - started,
            "history": rows,
            "nyu_metrics": scores,
        }
        torch.save(model.state_dict(), args.output / f"{name}.pt")
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        print(
            f"{name:10s} coupled dz {scores['coupled']['delta_z']:.4f} "
            f"dq {scores['coupled']['delta_q_rad']:.4f} | conditional dz "
            f"{scores['conditional']['delta_z']:.4f} "
            f"dq {scores['conditional']['delta_q_rad']:.4f}",
            flush=True,
        )
    print(f"results written to {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
