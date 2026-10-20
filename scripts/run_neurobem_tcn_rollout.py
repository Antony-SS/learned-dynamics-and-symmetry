"""Retrain NeuroBEM TCN-medium and evaluate a genuine open-loop rollout."""

from __future__ import annotations

import argparse
import bisect
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from quadrotor.geometry import quat_to_matrix
from quadrotor.neurobem_data import DEFAULT_ROOT
from quadrotor.neurobem_tcn import NeuroBEMTCN
from quadrotor.train import _apply, _so3_log

MASS = 0.772
GRAVITY = 9.80665
INERTIA = np.asarray((0.0025, 0.0021, 0.0043), dtype=np.float32)


@dataclass
class TCNFlight:
    name: str
    features: torch.Tensor
    labels: torch.Tensor
    position: torch.Tensor
    rotation: torch.Tensor
    velocity_world: torch.Tensor
    omega: torch.Tensor
    motors: torch.Tensor
    voltage: torch.Tensor
    dt: torch.Tensor


def _segment_id(path: Path) -> str:
    return path.stem.removeprefix("merged_")


def _flight_id(path: Path) -> str:
    return _segment_id(path).split("_seg_", 1)[0]


def _load(path: Path) -> TCNFlight:
    frame = pd.read_csv(path)
    omega = frame[["ang vel x", "ang vel y", "ang vel z"]].to_numpy(np.float32)
    velocity_body = frame[["vel x", "vel y", "vel z"]].to_numpy(np.float32)
    motors = frame[["mot 1", "mot 2", "mot 3", "mot 4"]].to_numpy(np.float32)
    features = np.concatenate((omega, velocity_body, motors), axis=1)
    acceleration = frame[["acc x", "acc y", "acc z"]].to_numpy(np.float32)
    angular_acceleration = frame[
        ["ang acc x", "ang acc y", "ang acc z"]
    ].to_numpy(np.float32)
    angular_momentum = omega * INERTIA
    force = MASS * acceleration
    torque = angular_acceleration * INERTIA + np.cross(omega, angular_momentum)
    quaternion = frame[["quat x", "quat y", "quat z", "quat w"]].to_numpy(
        np.float32
    )
    rotation = quat_to_matrix(quaternion).astype(np.float32)
    velocity_world = np.einsum("nij,nj->ni", rotation, velocity_body).astype(
        np.float32
    )
    time_values = frame["t"].to_numpy(np.float64)
    return TCNFlight(
        name=_segment_id(path),
        features=torch.from_numpy(features),
        labels=torch.from_numpy(np.concatenate((force, torque), axis=1)),
        position=torch.from_numpy(
            frame[["pos x", "pos y", "pos z"]].to_numpy(np.float32)
        ),
        rotation=torch.from_numpy(rotation),
        velocity_world=torch.from_numpy(velocity_world),
        omega=torch.from_numpy(omega),
        motors=torch.from_numpy(motors),
        voltage=torch.from_numpy(frame["vbat"].to_numpy(np.float32)),
        dt=torch.from_numpy(np.diff(time_values).astype(np.float32)),
    )


def _load_split(root: Path, seed: int) -> tuple[list[TCNFlight], ...]:
    paths = sorted((root / "processed_data").glob("merged_*.csv"))
    test_ids = {
        line.strip()
        for line in (root / "testset.txt").read_text().splitlines()
        if line.strip()
    }
    test_paths = [path for path in paths if _segment_id(path) in test_ids]
    remaining = [path for path in paths if _segment_id(path) not in test_ids]
    by_flight: dict[str, list[Path]] = {}
    for path in remaining:
        by_flight.setdefault(_flight_id(path), []).append(path)
    flight_ids = sorted(by_flight)
    shuffled = [
        flight_ids[index]
        for index in np.random.default_rng(seed).permutation(len(flight_ids))
    ]
    n_validation = max(1, int(round(0.20 * len(shuffled))))
    validation_ids = set(shuffled[:n_validation])
    validation_paths = [
        path for flight in validation_ids for path in sorted(by_flight[flight])
    ]
    train_paths = [
        path
        for flight in shuffled[n_validation:]
        for path in sorted(by_flight[flight])
    ]
    return (
        [_load(path) for path in train_paths],
        [_load(path) for path in validation_paths],
        [_load(path) for path in test_paths],
    )


class FlightWindows(Dataset):
    def __init__(self, flights: list[TCNFlight], stride: int) -> None:
        self.flights = flights
        self.stride = stride
        self.counts = [
            max(0, (len(flight.features) - NeuroBEMTCN.history_steps) // stride + 1)
            for flight in flights
        ]
        self.ends = np.cumsum(self.counts).tolist()

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        flight_index = bisect.bisect_right(self.ends, index)
        previous = 0 if flight_index == 0 else self.ends[flight_index - 1]
        target = NeuroBEMTCN.history_steps - 1 + (index - previous) * self.stride
        flight = self.flights[flight_index]
        history = flight.features[
            target - NeuroBEMTCN.history_steps + 1 : target + 1
        ]
        return history, flight.labels[target]


def _statistics(flights: list[TCNFlight]) -> tuple[torch.Tensor, ...]:
    inputs = torch.cat([flight.features for flight in flights])
    outputs = torch.cat([flight.labels for flight in flights])
    return (
        inputs.mean(0),
        inputs.std(0),
        outputs.mean(0),
        outputs.std(0),
    )


def _loss(model: NeuroBEMTCN, history: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target = (labels - model.output_center) / model.output_scale
    error = model.normalized_prediction(history) - target
    weights = error.new_tensor((1.0, 1.0, 3.0, 1.0, 1.0, 1.0))
    return (error.square() * weights).sum(dim=-1).mean()


def _mean_loss(
    model: NeuroBEMTCN, loader: DataLoader, device: torch.device
) -> float:
    total = 0.0
    seen = 0
    model.eval()
    with torch.inference_mode():
        for history, labels in loader:
            history, labels = history.to(device), labels.to(device)
            total += float(_loss(model, history, labels)) * len(history)
            seen += len(history)
    return total / seen


def _one_step_metrics(
    model: NeuroBEMTCN,
    flights: list[TCNFlight],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    loader = DataLoader(FlightWindows(flights, 1), batch_size=batch_size)
    errors = []
    model.eval()
    with torch.inference_mode():
        for history, labels in loader:
            prediction = model(history.to(device)).cpu()
            errors.append(prediction - labels)
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
    model: NeuroBEMTCN,
    flights: list[TCNFlight],
    device: torch.device,
    *,
    horizon: int,
    start_stride: int,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    starts = [
        (flight_index, start)
        for flight_index, flight in enumerate(flights)
        for start in range(
            NeuroBEMTCN.history_steps - 1,
            len(flight.features) - horizon,
            start_stride,
        )
    ]
    totals = {
        name: torch.zeros(horizon, dtype=torch.float64)
        for name in ("position", "velocity", "rotation", "omega")
    }
    inertia = torch.as_tensor(INERTIA, device=device)
    up_world = torch.tensor((0.0, 0.0, 1.0), device=device)
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            batch = starts[offset : offset + batch_size]
            histories, positions, rotations, velocities, omegas = [], [], [], [], []
            for flight_index, start in batch:
                flight = flights[flight_index]
                histories.append(
                    flight.features[
                        start - NeuroBEMTCN.history_steps + 1 : start + 1
                    ]
                )
                positions.append(flight.position[start])
                rotations.append(flight.rotation[start])
                velocities.append(flight.velocity_world[start])
                omegas.append(flight.omega[start])
            history = torch.stack(histories).to(device)
            position = torch.stack(positions).to(device)
            rotation = torch.stack(rotations).to(device)
            velocity = torch.stack(velocities).to(device)
            omega = torch.stack(omegas).to(device)

            for step in range(horizon):
                force_torque = model(history)
                force, torque = force_torque[:, :3], force_torque[:, 3:]
                gravity_body = torch.bmm(
                    rotation.transpose(1, 2),
                    up_world.expand(len(batch), 3).unsqueeze(-1),
                ).squeeze(-1)
                linear_acceleration = force / MASS - GRAVITY * gravity_body
                angular_momentum = omega * inertia
                angular_acceleration = (
                    torque - torch.linalg.cross(omega, angular_momentum)
                ) / inertia
                dt = torch.stack(
                    [
                        flights[flight_index].dt[start + step]
                        for flight_index, start in batch
                    ]
                ).to(device)
                position, rotation, velocity, omega = _apply(
                    position,
                    rotation,
                    velocity,
                    omega,
                    torch.cat((linear_acceleration, angular_acceleration), dim=-1),
                    dt,
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
                values = {
                    "position": torch.linalg.vector_norm(
                        position - true_position, dim=-1
                    ),
                    "velocity": torch.linalg.vector_norm(
                        velocity - true_velocity, dim=-1
                    ),
                    "rotation": torch.linalg.vector_norm(_so3_log(relative), dim=-1),
                    "omega": torch.linalg.vector_norm(omega - true_omega, dim=-1),
                }
                for name, value in values.items():
                    totals[name][step] += value.sum().cpu().double()

                velocity_body = torch.bmm(
                    rotation.transpose(1, 2), velocity.unsqueeze(-1)
                ).squeeze(-1)
                motors = torch.stack(
                    [
                        flights[flight_index].motors[start + target_index]
                        for flight_index, start in batch
                    ]
                ).to(device)
                next_features = torch.cat((omega, velocity_body, motors), dim=-1)
                history = torch.cat((history[:, 1:], next_features.unsqueeze(1)), dim=1)

    curves = {name: values / len(starts) for name, values in totals.items()}
    return {
        name: {
            "h10_0.025s": float(values[9]),
            "h20_0.05s": float(values[19]),
            "h40_0.1s": float(values[39]),
            "h100_0.25s": float(values[99]),
            "h200_0.5s": float(values[199]),
            "sum_0.5s_at_400hz": float(values.sum()),
        }
        for name, values in curves.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("runs/neurobem_tcn_medium"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--target-stride", type=int, default=4)
    parser.add_argument("--rollout-start-stride", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    train, validation, test = _load_split(args.data, args.seed)
    model = NeuroBEMTCN().to(device)
    model.set_normalization(
        *[value.to(device) for value in _statistics(train)]
    )
    train_loader = DataLoader(
        FlightWindows(train, args.target_stride),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        FlightWindows(validation, args.target_stride),
        batch_size=args.batch_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history_records = []
    started = time.perf_counter()
    for epoch in range(args.epochs + 1):
        if epoch:
            model.train()
            total = 0.0
            seen = 0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                loss = _loss(model, features, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(features)
                seen += len(features)
            train_loss = total / seen
            scheduler.step()
        else:
            train_loss = _mean_loss(model, train_loader, device)
        validation_loss = _mean_loss(model, validation_loader, device)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        history_records.append(
            {
                "epoch": epoch,
                "train_normalized_weighted_mse": train_loss,
                "validation_normalized_weighted_mse": validation_loss,
            }
        )
        print(
            f"epoch {epoch:03d} train {train_loss:.6f} "
            f"val {validation_loss:.6f}"
            f"{' saved' if epoch == best_epoch else ''}",
            flush=True,
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.output / "tcn_medium_none_nn.pt")
    one_step = _one_step_metrics(model, test, device, args.batch_size)
    rollout = _rollout_metrics(
        model,
        test,
        device,
        horizon=200,
        start_stride=args.rollout_start_stride,
        batch_size=256,
    )
    results = {
        "protocol": {
            "model": "NeuroBEM TCN-medium architecture, pure learned None+NN",
            "training_objective": "paper one-step normalized force/torque loss",
            "rollout": "seed once after 50 ms true history, then fully autoregressive state; measured motors only",
            "source_rate_hz": 400,
            "history_steps": 20,
            "history_seconds": 0.05,
            "rollout_horizon_steps": 200,
            "rollout_horizon_seconds": 0.5,
            "rollout_start_stride": args.rollout_start_stride,
            "official_test_segments": len(test),
            "device": str(device),
        },
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "training_seconds": time.perf_counter() - started,
            "history": history_records,
            "test_one_step": one_step,
            "test_rollout_mae": rollout,
        },
        "paper_none_nn_reference": {
            "force_rmse_N": 0.438,
            "torque_rmse_Nm": 0.014,
        },
    }
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"one_step": one_step, "rollout": rollout}, indent=2))


if __name__ == "__main__":
    main()
