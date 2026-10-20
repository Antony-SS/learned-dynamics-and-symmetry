"""Evaluate NeuroBEM checkpoints with the NYU 60-step metrics.

Two protocols are reported:

``coupled``
    Position, attitude, velocity, and angular velocity all propagate from the
    model's predictions.

``conditional``
    Matches the best decoupled protocol in Rao et al. (IROS 2024). The
    velocity rollout receives ground-truth future attitude, while the separate
    attitude rollout receives ground-truth future velocity and angular
    velocity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from quadrotor.neurobem_data import load_neurobem_prepared
from quadrotor.train import _apply, _features, _so3_log
from run_neurobem_gru import AbsoluteModel, CanonicalModel, SymmetricModel


def _starts(flights: list, history: int, horizon: int) -> list[tuple[int, int]]:
    return [
        (flight_index, start)
        for flight_index, flight in enumerate(flights)
        for start in range(history - 1, flight.n_steps - horizon + 1)
    ]


def _batch(
    flights: list,
    pairs: list[tuple[int, int]],
    history: int,
    horizon: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    pwm, voltage, dt, position, rotation, velocity, omega = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    past_position, past_rotation, past_velocity, past_omega, past_pwm, past_voltage = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for flight_index, start in pairs:
        flight = flights[flight_index]
        stop = start + horizon
        history_start = start - history + 1
        pwm.append(flight.pwm[start:stop])
        voltage.append(flight.voltage[start:stop])
        dt.append(flight.dt[start:stop])
        position.append(flight.position[start : stop + 1])
        rotation.append(flight.rotation[start : stop + 1])
        velocity.append(flight.velocity_world[start : stop + 1])
        omega.append(flight.omega_body[start : stop + 1])
        past_position.append(flight.position[history_start:start])
        past_rotation.append(flight.rotation[history_start:start])
        past_velocity.append(flight.velocity_world[history_start:start])
        past_omega.append(flight.omega_body[history_start:start])
        past_pwm.append(flight.pwm[history_start:start])
        past_voltage.append(flight.voltage[history_start:start])
    return tuple(
        torch.as_tensor(np.stack(values), device=device)
        for values in (
            pwm,
            voltage,
            dt,
            position,
            rotation,
            velocity,
            omega,
            past_position,
            past_rotation,
            past_velocity,
            past_omega,
            past_pwm,
            past_voltage,
        )
    )


def _warm_hidden(
    model: torch.nn.Module,
    position: torch.Tensor,
    rotation: torch.Tensor,
    velocity: torch.Tensor,
    omega: torch.Tensor,
    pwm: torch.Tensor,
    voltage: torch.Tensor,
) -> torch.Tensor:
    hidden = model.initial_hidden(position.shape[0], position)
    for step in range(pwm.shape[1]):
        features = _features(
            model,
            position[:, step],
            rotation[:, step],
            velocity[:, step],
            omega[:, step],
            pwm[:, step],
            voltage[:, step],
        )
        _, hidden = model.step(features, hidden)
    return hidden


def _coupled(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    pwm: torch.Tensor,
    voltage: torch.Tensor,
    dt: torch.Tensor,
    position: torch.Tensor,
    rotation: torch.Tensor,
    velocity: torch.Tensor,
    omega: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotations, velocities, omegas = [], [], []
    for step in range(pwm.shape[1]):
        features = _features(
            model,
            position,
            rotation,
            velocity,
            omega,
            pwm[:, step],
            voltage[:, step],
        )
        acceleration, hidden = model.step(features, hidden)
        position, rotation, velocity, omega = _apply(
            position, rotation, velocity, omega, acceleration, dt[:, step]
        )
        rotations.append(rotation)
        velocities.append(velocity)
        omegas.append(omega)
    return (
        torch.stack(rotations, dim=1),
        torch.stack(velocities, dim=1),
        torch.stack(omegas, dim=1),
    )


def _conditional_velocity(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    pwm: torch.Tensor,
    voltage: torch.Tensor,
    dt: torch.Tensor,
    position_gt: torch.Tensor,
    rotation_gt: torch.Tensor,
    velocity_gt: torch.Tensor,
    omega_gt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    position = position_gt[:, 0]
    velocity = velocity_gt[:, 0]
    omega = omega_gt[:, 0]
    velocities, omegas = [], []
    for step in range(pwm.shape[1]):
        # Rao et al.'s velocity module receives the true attitude at every
        # future step while its velocity channels remain autoregressive.
        rotation = rotation_gt[:, step]
        features = _features(
            model,
            position,
            rotation,
            velocity,
            omega,
            pwm[:, step],
            voltage[:, step],
        )
        acceleration, hidden = model.step(features, hidden)
        position, _, velocity, omega = _apply(
            position, rotation, velocity, omega, acceleration, dt[:, step]
        )
        velocities.append(velocity)
        omegas.append(omega)
    return torch.stack(velocities, dim=1), torch.stack(omegas, dim=1)


def _conditional_attitude(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    pwm: torch.Tensor,
    voltage: torch.Tensor,
    dt: torch.Tensor,
    position_gt: torch.Tensor,
    rotation_gt: torch.Tensor,
    velocity_gt: torch.Tensor,
    omega_gt: torch.Tensor,
) -> torch.Tensor:
    position = position_gt[:, 0]
    rotation = rotation_gt[:, 0]
    rotations = []
    for step in range(pwm.shape[1]):
        # Rao et al.'s attitude module receives true future linear and angular
        # velocity while only attitude remains autoregressive.
        velocity = velocity_gt[:, step]
        omega = omega_gt[:, step]
        features = _features(
            model,
            position,
            rotation,
            velocity,
            omega,
            pwm[:, step],
            voltage[:, step],
        )
        acceleration, hidden = model.step(features, hidden)
        position, rotation, _, _ = _apply(
            position, rotation, velocity, omega, acceleration, dt[:, step]
        )
        rotations.append(rotation)
    return torch.stack(rotations, dim=1)


def _errors(
    velocity: torch.Tensor,
    omega: torch.Tensor,
    rotation: torch.Tensor,
    velocity_gt: torch.Tensor,
    omega_gt: torch.Tensor,
    rotation_gt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    velocity_state_squared = (
        (velocity - velocity_gt[:, 1:]).square().sum(dim=-1)
        + (omega - omega_gt[:, 1:]).square().sum(dim=-1)
    )
    relative = torch.matmul(rotation_gt[:, 1:].transpose(-1, -2), rotation)
    attitude_angle = torch.linalg.vector_norm(_so3_log(relative), dim=-1)
    return velocity_state_squared, attitude_angle


def evaluate(
    model: torch.nn.Module,
    flights: list,
    *,
    history: int,
    horizon: int,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    pairs = _starts(flights, history, horizon)
    device = next(model.parameters()).device
    totals = {
        protocol: {
            "velocity_state": torch.zeros((), dtype=torch.float64),
            "attitude": torch.zeros((), dtype=torch.float64),
        }
        for protocol in ("coupled", "conditional")
    }
    count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            values = _batch(
                flights,
                pairs[start : start + batch_size],
                history,
                horizon,
                device,
            )
            (
                pwm,
                voltage,
                dt,
                position_gt,
                rotation_gt,
                velocity_gt,
                omega_gt,
                past_position,
                past_rotation,
                past_velocity,
                past_omega,
                past_pwm,
                past_voltage,
            ) = values
            hidden = _warm_hidden(
                model,
                past_position,
                past_rotation,
                past_velocity,
                past_omega,
                past_pwm,
                past_voltage,
            )
            rotation, velocity, omega = _coupled(
                model,
                hidden.clone(),
                pwm,
                voltage,
                dt,
                position_gt[:, 0],
                rotation_gt[:, 0],
                velocity_gt[:, 0],
                omega_gt[:, 0],
            )
            squared, angle = _errors(
                velocity,
                omega,
                rotation,
                velocity_gt,
                omega_gt,
                rotation_gt,
            )
            totals["coupled"]["velocity_state"] += squared.sum().cpu().double()
            totals["coupled"]["attitude"] += angle.sum().cpu().double()

            velocity, omega = _conditional_velocity(
                model,
                hidden.clone(),
                pwm,
                voltage,
                dt,
                position_gt,
                rotation_gt,
                velocity_gt,
                omega_gt,
            )
            rotation = _conditional_attitude(
                model,
                hidden.clone(),
                pwm,
                voltage,
                dt,
                position_gt,
                rotation_gt,
                velocity_gt,
                omega_gt,
            )
            squared, angle = _errors(
                velocity,
                omega,
                rotation,
                velocity_gt,
                omega_gt,
                rotation_gt,
            )
            totals["conditional"]["velocity_state"] += squared.sum().cpu().double()
            totals["conditional"]["attitude"] += angle.sum().cpu().double()
            count += squared.numel()
    return {
        protocol: {
            "delta_z": float(values["velocity_state"] / count),
            "delta_q_rad": float(values["attitude"] / count),
        }
        for protocol, values in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/neurobem_gru_20ep"))
    parser.add_argument("--history", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _, _, test_flights = load_neurobem_prepared(seed=args.seed)
    constructors = {
        "absolute": AbsoluteModel,
        "canonical": CanonicalModel,
        "symmetric": SymmetricModel,
    }
    results: dict[str, object] = {
        "protocol": {
            "source": "Rao et al., Learning Long-Horizon Predictions for Quadrotor Dynamics, IROS 2024",
            "horizon_steps": args.horizon,
            "horizon_seconds": args.horizon / 100.0,
            "history_steps": args.history,
            "history_seconds": args.history / 100.0,
            "test_split": "NeuroBEM released testset.txt (13 segments; NYU reports 12)",
            "delta_z": "mean_t(||v-vhat||^2 + ||omega-omegahat||^2)",
            "delta_q": "mean_t geodesic SO(3) attitude error",
        },
        "paper_decoupled_tcn": {"delta_z": 0.042, "delta_q_rad": 0.006},
        "paper_decoupled_gru": {"delta_z": 0.090, "delta_q_rad": 0.010},
        "models": {},
    }
    for name, constructor in constructors.items():
        model = constructor()
        model.load_state_dict(
            torch.load(args.run / f"{name}.pt", map_location="cpu", weights_only=True)
        )
        scores = evaluate(
            model,
            test_flights,
            history=args.history,
            horizon=args.horizon,
            batch_size=args.batch_size,
        )
        results["models"][name] = scores
        print(
            f"{name:10s} coupled dz {scores['coupled']['delta_z']:.4f} "
            f"dq {scores['coupled']['delta_q_rad']:.4f} rad | "
            f"conditional dz {scores['conditional']['delta_z']:.4f} "
            f"dq {scores['conditional']['delta_q_rad']:.4f} rad",
            flush=True,
        )
    output = args.run / "nyu_protocol_results.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"results written to {output}")


if __name__ == "__main__":
    main()
