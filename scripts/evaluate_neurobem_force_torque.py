"""Evaluate trained rollout models using NeuroBEM's force/torque RMSE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quadrotor.neurobem_data import DEFAULT_ROOT, load_neurobem_prepared
from quadrotor.train import _features
from run_neurobem_gru import AbsoluteModel, CanonicalModel, SymmetricModel

MASS = 0.772
GRAVITY = 9.80665
INERTIA = np.asarray((0.0025, 0.0021, 0.0043), dtype=np.float32)


def _rmse(error: np.ndarray) -> dict[str, float]:
    return {
        "xy": float(np.sqrt(np.mean(error[:, :2] ** 2))),
        "z": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "all": float(np.sqrt(np.mean(error**2))),
    }


def _labels(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(root / "processed_data" / f"merged_{name}.csv").iloc[::4]
    force = MASS * frame[["acc x", "acc y", "acc z"]].to_numpy(dtype=np.float32)
    omega = frame[["ang vel x", "ang vel y", "ang vel z"]].to_numpy(dtype=np.float32)
    alpha = frame[["ang acc x", "ang acc y", "ang acc z"]].to_numpy(dtype=np.float32)
    angular_momentum = omega * INERTIA
    torque = alpha * INERTIA + np.cross(omega, angular_momentum)
    return force, torque


def _predict(
    model: torch.nn.Module,
    flights: list,
    root: Path,
    *,
    history: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predicted_force, true_force, predicted_torque, true_torque = [], [], [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for flight in flights:
            position = torch.as_tensor(flight.position[:-1], device=device)
            rotation = torch.as_tensor(flight.rotation[:-1], device=device)
            velocity = torch.as_tensor(flight.velocity_world[:-1], device=device)
            omega = torch.as_tensor(flight.omega_body[:-1], device=device)
            pwm = torch.as_tensor(flight.pwm, device=device)
            voltage = torch.as_tensor(flight.voltage, device=device)
            features = _features(
                model, position, rotation, velocity, omega, pwm, voltage
            )
            force_target, torque_target = _labels(root, flight.name)
            force_target = force_target[: flight.n_steps]
            torque_target = torque_target[: flight.n_steps]

            targets = torch.arange(history - 1, flight.n_steps, device=device)
            for start in range(0, targets.numel(), batch_size):
                selected = targets[start : start + batch_size]
                hidden = model.initial_hidden(selected.numel(), features)
                acceleration = None
                for offset in range(history - 1, -1, -1):
                    acceleration, hidden = model.step(
                        features[selected - offset], hidden
                    )
                assert acceleration is not None
                up_body = torch.bmm(
                    rotation[selected].transpose(1, 2),
                    rotation.new_tensor((0.0, 0.0, 1.0))
                    .expand(selected.numel(), 3)
                    .unsqueeze(-1),
                ).squeeze(-1)
                force = MASS * (acceleration[:, :3] + GRAVITY * up_body)
                angular_momentum = omega[selected] * torch.as_tensor(
                    INERTIA, device=device
                )
                torque = (
                    acceleration[:, 3:] * torch.as_tensor(INERTIA, device=device)
                    + torch.linalg.cross(omega[selected], angular_momentum)
                )
                indices = selected.detach().cpu().numpy()
                predicted_force.append(force.cpu().numpy())
                true_force.append(force_target[indices])
                predicted_torque.append(torque.cpu().numpy())
                true_torque.append(torque_target[indices])
    return tuple(
        np.concatenate(values, axis=0)
        for values in (predicted_force, true_force, predicted_torque, true_torque)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/neurobem_gru_20ep"))
    parser.add_argument("--data", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    _, _, test_flights = load_neurobem_prepared(args.data)
    constructors = {
        "absolute": AbsoluteModel,
        "canonical": CanonicalModel,
        "symmetric": SymmetricModel,
    }
    results: dict[str, object] = {
        "protocol": {
            "test_split": "released testset.txt (13 segments)",
            "rate_hz": 100,
            "history_steps": args.history,
            "history_seconds": args.history / 100.0,
            "force_definition": "m * (predicted kinematic acceleration - R^T gravity)",
            "torque_definition": "J * predicted angular acceleration + omega x J omega",
            "rmse_definition": "sqrt(mean(componentwise squared error))",
        },
        "paper_references": {
            "None+NN": {"force_N": 0.438, "torque_Nm": 0.014},
            "BEM+NN": {"force_N": 0.335, "torque_Nm": 0.012},
            "HDVIO": {"force_N": 0.491},
            "HDVIO2.0": {"force_N": 0.491, "torque_Nm": 0.012},
        },
        "models": {},
    }
    for name, constructor in constructors.items():
        model = constructor()
        model.load_state_dict(
            torch.load(args.run / f"{name}.pt", map_location="cpu", weights_only=True)
        )
        predicted_force, true_force, predicted_torque, true_torque = _predict(
            model,
            test_flights,
            args.data,
            history=args.history,
            batch_size=args.batch_size,
        )
        force = _rmse(predicted_force - true_force)
        torque = _rmse(predicted_torque - true_torque)
        results["models"][name] = {
            "samples": int(predicted_force.shape[0]),
            "force_rmse_N": force,
            "torque_rmse_Nm": torque,
        }
        print(
            f"{name:10s} force {force['all']:.4f} N "
            f"(xy {force['xy']:.4f}, z {force['z']:.4f})  "
            f"torque {torque['all']:.5f} Nm "
            f"(xy {torque['xy']:.5f}, z {torque['z']:.5f})",
            flush=True,
        )
    output = args.run / "force_torque_results.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"results written to {output}")


if __name__ == "__main__":
    main()
