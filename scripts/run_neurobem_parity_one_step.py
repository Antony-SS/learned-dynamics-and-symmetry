"""Train the compact parity recurrent model on NeuroBEM's one-step objective."""

from __future__ import annotations

import argparse
import bisect
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from quadrotor.canonical import canonical_dim, output_dim
from quadrotor.neurobem_data import DEFAULT_ROOT
from quadrotor.sequence import ParityMotorNet
from quadrotor.train import _apply, _so3_log
try:
    from scripts.run_neurobem_tcn_rollout import (
        GRAVITY,
        INERTIA,
        MASS,
        TCNFlight,
        _load_split,
    )
except ModuleNotFoundError:
    from run_neurobem_tcn_rollout import (
        GRAVITY,
        INERTIA,
        MASS,
        TCNFlight,
        _load_split,
    )

HISTORY = 20


def _canonical_features(flight: TCNFlight) -> torch.Tensor:
    up_world = flight.rotation.new_tensor((0.0, 0.0, 1.0)).expand(
        len(flight.rotation), 3
    )
    gravity_body = torch.bmm(
        flight.rotation.transpose(1, 2), up_world.unsqueeze(-1)
    ).squeeze(-1)
    omega = flight.features[:, :3]
    velocity_body = flight.features[:, 3:6]
    # Released source order is [BR, FR, BL, FL]; symmetry code uses
    # [FR, BR, BL, FL].
    motors = flight.motors[:, [1, 0, 2, 3]]
    return torch.cat(
        (
            gravity_body,
            velocity_body,
            omega,
            motors,
            flight.voltage.unsqueeze(-1),
        ),
        dim=-1,
    )


class ParityWindows(Dataset):
    def __init__(self, flights: list[TCNFlight], stride: int) -> None:
        self.features = [_canonical_features(flight) for flight in flights]
        self.labels = [flight.labels for flight in flights]
        self.stride = stride
        self.counts = [
            max(0, (len(features) - HISTORY) // stride + 1)
            for features in self.features
        ]
        self.ends = np.cumsum(self.counts).tolist()

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        flight_index = bisect.bisect_right(self.ends, index)
        previous = 0 if flight_index == 0 else self.ends[flight_index - 1]
        target = HISTORY - 1 + (index - previous) * self.stride
        return (
            self.features[flight_index][target - HISTORY + 1 : target + 1],
            self.labels[flight_index][target],
        )


def _set_input_normalization(
    model: ParityMotorNet, features: list[torch.Tensor]
) -> None:
    rows = torch.cat(features)
    center = rows.mean(0)
    scale = rows.std(0)
    center[0:2] = 0.0
    center[3:9] = 0.0
    scale[0:2] = rows[:, 0:2].square().mean().sqrt()
    scale[3:5] = rows[:, 3:5].square().mean().sqrt()
    scale[6:8] = rows[:, 6:8].square().mean().sqrt()
    scale[8] = rows[:, 8].square().mean().sqrt()
    center[9:13] = rows[:, 9:13].mean()
    scale[9:13] = rows[:, 9:13].std()
    model.set_input_normalization(center, scale)


def _prediction(
    model: ParityMotorNet, history: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = model.initial_hidden(len(history), history)
    acceleration = None
    for step in range(history.shape[1]):
        acceleration, hidden = model.step(history[:, step], hidden)
    assert acceleration is not None
    gravity = history[:, -1, :3]
    omega = history[:, -1, 6:9]
    inertia = acceleration.new_tensor(INERTIA)
    force = MASS * (acceleration[:, :3] + GRAVITY * gravity)
    torque = (
        acceleration[:, 3:] * inertia
        + torch.linalg.cross(omega, omega * inertia)
    )
    return torch.cat((force, torque), dim=-1), hidden


def _loss(
    model: ParityMotorNet,
    history: torch.Tensor,
    labels: torch.Tensor,
    output_center: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    prediction, _ = _prediction(model, history)
    error = (prediction - labels) / output_scale
    weights = error.new_tensor((1.0, 1.0, 3.0, 1.0, 1.0, 1.0))
    return (error.square() * weights).sum(dim=-1).mean()


def _mean_loss(
    model: ParityMotorNet,
    loader: DataLoader,
    device: torch.device,
    output_center: torch.Tensor,
    output_scale: torch.Tensor,
) -> float:
    del output_center
    total = 0.0
    seen = 0
    model.eval()
    with torch.inference_mode():
        for history, labels in loader:
            history, labels = history.to(device), labels.to(device)
            value = _loss(
                model,
                history,
                labels,
                torch.empty(0, device=device),
                output_scale,
            )
            total += float(value) * len(history)
            seen += len(history)
    return total / seen


def _one_step_metrics(
    model: ParityMotorNet,
    dataset: ParityWindows,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    errors = []
    model.eval()
    with torch.inference_mode():
        for history, labels in DataLoader(dataset, batch_size=batch_size):
            prediction, _ = _prediction(model, history.to(device))
            errors.append(prediction.cpu() - labels)
    error = torch.cat(errors)
    return {
        "force_rmse_N": float(error[:, :3].square().mean().sqrt()),
        "torque_rmse_Nm": float(error[:, 3:].square().mean().sqrt()),
        "force_xy_rmse_N": float(error[:, :2].square().mean().sqrt()),
        "force_z_rmse_N": float(error[:, 2].square().mean().sqrt()),
        "torque_xy_rmse_Nm": float(error[:, 3:5].square().mean().sqrt()),
        "torque_z_rmse_Nm": float(error[:, 5].square().mean().sqrt()),
    }


def _rollout_metrics(
    model: ParityMotorNet,
    flights: list[TCNFlight],
    device: torch.device,
    *,
    horizon: int,
    start_stride: int,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    all_features = [_canonical_features(flight) for flight in flights]
    starts = [
        (flight_index, start)
        for flight_index, flight in enumerate(flights)
        for start in range(HISTORY - 1, len(flight.features) - horizon, start_stride)
    ]
    totals = {
        name: torch.zeros(horizon, dtype=torch.float64)
        for name in ("position", "velocity", "rotation", "omega")
    }
    for offset in range(0, len(starts), batch_size):
        batch = starts[offset : offset + batch_size]
        histories, positions, rotations, velocities, omegas = [], [], [], [], []
        for flight_index, start in batch:
            histories.append(all_features[flight_index][start - HISTORY + 1 : start + 1])
            flight = flights[flight_index]
            positions.append(flight.position[start])
            rotations.append(flight.rotation[start])
            velocities.append(flight.velocity_world[start])
            omegas.append(flight.omega[start])
        history = torch.stack(histories).to(device)
        position = torch.stack(positions).to(device)
        rotation = torch.stack(rotations).to(device)
        velocity = torch.stack(velocities).to(device)
        omega = torch.stack(omegas).to(device)
        with torch.inference_mode():
            acceleration, hidden = None, model.initial_hidden(len(batch), history)
            for warm_step in range(HISTORY):
                acceleration, hidden = model.step(history[:, warm_step], hidden)
            assert acceleration is not None
            for step in range(horizon):
                dt = torch.stack(
                    [
                        flights[flight_index].dt[start + step]
                        for flight_index, start in batch
                    ]
                ).to(device)
                position, rotation, velocity, omega = _apply(
                    position, rotation, velocity, omega, acceleration, dt
                )
                target_index = step + 1
                true_position = torch.stack(
                    [
                        flights[flight_index].position[start + target_index]
                        for flight_index, start in batch
                    ]
                ).to(device)
                true_rotation = torch.stack(
                    [
                        flights[flight_index].rotation[start + target_index]
                        for flight_index, start in batch
                    ]
                ).to(device)
                true_velocity = torch.stack(
                    [
                        flights[flight_index].velocity_world[start + target_index]
                        for flight_index, start in batch
                    ]
                ).to(device)
                true_omega = torch.stack(
                    [
                        flights[flight_index].omega[start + target_index]
                        for flight_index, start in batch
                    ]
                ).to(device)
                relative = torch.bmm(true_rotation.transpose(1, 2), rotation)
                errors = {
                    "position": torch.linalg.vector_norm(position - true_position, dim=-1),
                    "velocity": torch.linalg.vector_norm(velocity - true_velocity, dim=-1),
                    "rotation": torch.linalg.vector_norm(_so3_log(relative), dim=-1),
                    "omega": torch.linalg.vector_norm(omega - true_omega, dim=-1),
                }
                for name, error in errors.items():
                    totals[name][step] += error.sum().cpu().double()

                velocity_body = torch.bmm(
                    rotation.transpose(1, 2), velocity.unsqueeze(-1)
                ).squeeze(-1)
                up_world = rotation.new_tensor((0.0, 0.0, 1.0)).expand(len(batch), 3)
                gravity_body = torch.bmm(
                    rotation.transpose(1, 2), up_world.unsqueeze(-1)
                ).squeeze(-1)
                motors = torch.stack(
                    [
                        flights[flight_index].motors[start + target_index][[1, 0, 2, 3]]
                        for flight_index, start in batch
                    ]
                ).to(device)
                voltage = torch.stack(
                    [
                        flights[flight_index].voltage[start + target_index]
                        for flight_index, start in batch
                    ]
                ).to(device)
                features = torch.cat(
                    (
                        gravity_body,
                        velocity_body,
                        omega,
                        motors,
                        voltage.unsqueeze(-1),
                    ),
                    dim=-1,
                )
                acceleration, hidden = model.step(features, hidden)

    curves = {name: value / len(starts) for name, value in totals.items()}
    return {
        name: {
            "h10_0.025s": float(value[9]),
            "h20_0.05s": float(value[19]),
            "h40_0.1s": float(value[39]),
            "h100_0.25s": float(value[99]),
            "h200_0.5s": float(value[199]),
        }
        for name, value in curves.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/neurobem_parity_one_step")
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--target-stride", type=int, default=4)
    parser.add_argument("--rollout-start-stride", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps"), default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device
        or ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    train, validation, test = _load_split(args.data, args.seed)
    train_data = ParityWindows(train, args.target_stride)
    validation_data = ParityWindows(validation, args.target_stride)
    model = ParityMotorNet(
        canonical_dim(include_voltage=True), output_dim(), hidden_per_type=24
    ).to(device)
    model.reads_voltage = True
    _set_input_normalization(model, train_data.features)
    output_rows = torch.cat([flight.labels for flight in train])
    output_center = output_rows.mean(0).to(device)
    output_scale = output_rows.std(0).clamp_min(1e-6).to(device)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    records = []
    started = time.perf_counter()
    for epoch in range(args.epochs + 1):
        if epoch:
            model.train()
            total = 0.0
            seen = 0
            for history, labels in train_loader:
                history, labels = history.to(device), labels.to(device)
                loss = _loss(model, history, labels, output_center, output_scale)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(history)
                seen += len(history)
            train_loss = total / seen
            scheduler.step()
        else:
            train_loss = _mean_loss(
                model, train_loader, device, output_center, output_scale
            )
        validation_loss = _mean_loss(
            model, validation_loader, device, output_center, output_scale
        )
        saved = validation_loss < best_loss
        if saved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        records.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": validation_loss}
        )
        print(
            f"epoch {epoch:03d} train {train_loss:.6f} val {validation_loss:.6f}"
            f"{' saved' if saved else ''}",
            flush=True,
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.output / "parity_gru24_one_step.pt")
    one_step = _one_step_metrics(
        model, ParityWindows(test, 1), device, args.batch_size
    )
    rollout = _rollout_metrics(
        model,
        test,
        device,
        horizon=200,
        start_stride=args.rollout_start_stride,
        batch_size=256,
    )
    result = {
        "protocol": {
            "objective": "NeuroBEM one-step normalized force/torque",
            "rate_hz": 400,
            "history_steps": HISTORY,
            "history_seconds": 0.05,
            "rollout": "50 ms ground-truth warm start, then fully autoregressive state",
            "future_ground_truth_state_injection": False,
            "measured_motor_sequence": True,
        },
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "training_seconds": time.perf_counter() - started,
            "history": records,
            "test_one_step": one_step,
            "test_rollout_mae": rollout,
        },
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"one_step": one_step, "rollout": rollout}, indent=2))


if __name__ == "__main__":
    main()
