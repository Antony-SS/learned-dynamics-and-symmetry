"""Train parity GRU with one-step force/torque plus a short open-loop future."""

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
from quadrotor.sequence import ParityMotorNet, ResidualParityMotorNet
from quadrotor.train import _apply, _so3_log

try:
    from scripts.run_neurobem_parity_one_step import (
        HISTORY,
        _canonical_features,
        _one_step_metrics,
        _rollout_metrics,
        _set_input_normalization,
    )
    from scripts.run_neurobem_tcn_rollout import (
        GRAVITY,
        INERTIA,
        MASS,
        TCNFlight,
        _load_split,
    )
except ModuleNotFoundError:
    from run_neurobem_parity_one_step import (
        HISTORY,
        _canonical_features,
        _one_step_metrics,
        _rollout_metrics,
        _set_input_normalization,
    )
    from run_neurobem_tcn_rollout import (
        GRAVITY,
        INERTIA,
        MASS,
        TCNFlight,
        _load_split,
    )


class MixedWindows(Dataset):
    def __init__(self, flights: list[TCNFlight], horizon: int, stride: int) -> None:
        self.flights = flights
        self.features = [_canonical_features(flight) for flight in flights]
        self.horizon = horizon
        self.stride = stride
        self.counts = [
            max(0, (len(features) - HISTORY - horizon) // stride + 1)
            for features in self.features
        ]
        self.ends = np.cumsum(self.counts).tolist()

    def __len__(self) -> int:
        return int(self.ends[-1]) if self.ends else 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        flight_index = bisect.bisect_right(self.ends, index)
        previous = 0 if flight_index == 0 else self.ends[flight_index - 1]
        start = HISTORY - 1 + (index - previous) * self.stride
        flight = self.flights[flight_index]
        features = self.features[flight_index]
        stop = start + self.horizon
        return {
            "history": features[start - HISTORY + 1 : start + 1],
            "future_motors": flight.motors[start + 1 : stop + 1, [1, 0, 2, 3]],
            "future_voltage": flight.voltage[start + 1 : stop + 1],
            "labels": flight.labels[start],
            "position": flight.position[start : stop + 1],
            "rotation": flight.rotation[start : stop + 1],
            "velocity": flight.velocity_world[start : stop + 1],
            "omega": flight.omega[start : stop + 1],
            "dt": flight.dt[start:stop],
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in batch]) for key in batch[0]}


def _predicted_features(
    rotation: torch.Tensor,
    velocity: torch.Tensor,
    omega: torch.Tensor,
    motors: torch.Tensor,
    voltage: torch.Tensor,
) -> torch.Tensor:
    up_world = rotation.new_tensor((0.0, 0.0, 1.0)).expand(len(rotation), 3)
    gravity_body = torch.bmm(rotation.transpose(1, 2), up_world.unsqueeze(-1)).squeeze(-1)
    velocity_body = torch.bmm(
        rotation.transpose(1, 2), velocity.unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat(
        (gravity_body, velocity_body, omega, motors, voltage.unsqueeze(-1)),
        dim=-1,
    )


def _force_torque(acceleration: torch.Tensor, gravity: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    inertia = acceleration.new_tensor(INERTIA)
    force = MASS * (acceleration[:, :3] + GRAVITY * gravity)
    torque = acceleration[:, 3:] * inertia + torch.linalg.cross(omega, omega * inertia)
    return torch.cat((force, torque), dim=-1)


def _state_scales(flights: list[TCNFlight], device: torch.device) -> tuple[torch.Tensor, ...]:
    position = torch.cat([flight.position for flight in flights])
    velocity = torch.cat([flight.velocity_world for flight in flights])
    omega = torch.cat([flight.omega for flight in flights])
    rotation = _so3_log(torch.cat([flight.rotation for flight in flights]))
    values = (
        float(position.std()),
        float(rotation.std()),
        float(velocity.std()),
        float(omega.std()),
    )
    return tuple(
        torch.tensor(max(value, 1e-6), device=device) for value in values
    )


def _mixed_loss(
    model: ParityMotorNet,
    batch: dict[str, torch.Tensor],
    output_scale: torch.Tensor,
    scales: tuple[torch.Tensor, ...],
    one_step_weight: float,
    rollout_weight: float,
    residual_weight_l2: float = 0.0,
    residual_output_l2: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history = batch["history"]
    hidden = model.initial_hidden(len(history), history)
    acceleration = None
    residual_energy = history.new_zeros(())
    residual_steps = 0
    for step in range(history.shape[1]):
        acceleration, hidden = model.step(history[:, step], hidden)
        if getattr(model, "last_residual", None) is not None:
            residual_energy = residual_energy + model.last_residual.square().mean()
            residual_steps += 1
    assert acceleration is not None
    force_torque = _force_torque(acceleration, history[:, -1, :3], history[:, -1, 6:9])
    one_step = ((force_torque - batch["labels"]) / output_scale).square()
    axis_weight = one_step.new_tensor((1.0, 1.0, 3.0, 1.0, 1.0, 1.0))
    one_step_loss = (one_step * axis_weight).sum(dim=-1).mean()

    position = batch["position"][:, 0]
    rotation = batch["rotation"][:, 0]
    velocity = batch["velocity"][:, 0]
    omega = batch["omega"][:, 0]
    errors = []
    for step in range(batch["dt"].shape[1]):
        position, rotation, velocity, omega = _apply(
            position, rotation, velocity, omega, acceleration, batch["dt"][:, step]
        )
        relative = torch.bmm(batch["rotation"][:, step + 1].transpose(1, 2), rotation)
        errors.append(
            torch.cat(
                (
                    ((position - batch["position"][:, step + 1]) / scales[0]).square().mean(dim=-1, keepdim=True),
                    (_so3_log(relative) / scales[1]).square().mean(dim=-1, keepdim=True),
                    ((velocity - batch["velocity"][:, step + 1]) / scales[2]).square().mean(dim=-1, keepdim=True),
                    ((omega - batch["omega"][:, step + 1]) / scales[3]).square().mean(dim=-1, keepdim=True),
                ),
                dim=-1,
            )
        )
        acceleration, hidden = model.step(
            _predicted_features(
                rotation,
                velocity,
                omega,
                batch["future_motors"][:, step],
                batch["future_voltage"][:, step],
            ),
            hidden,
        )
        if getattr(model, "last_residual", None) is not None:
            residual_energy = residual_energy + model.last_residual.square().mean()
            residual_steps += 1
    rollout_loss = torch.stack(errors, dim=1).mean()
    if isinstance(model, ResidualParityMotorNet) and residual_steps:
        residual_loss = model.residual_penalty(
            residual_energy / residual_steps,
            residual_weight_l2,
            residual_output_l2,
        )
    else:
        residual_loss = one_step_loss.new_zeros(())
    return (
        one_step_weight * one_step_loss + rollout_weight * rollout_loss + residual_loss,
        one_step_loss,
        rollout_loss,
        residual_loss,
    )


def _mean_mixed(
    model: ParityMotorNet,
    loader: DataLoader,
    output_scale: torch.Tensor,
    scales: tuple[torch.Tensor, ...],
    one_step_weight: float,
    rollout_weight: float,
    residual_weight_l2: float = 0.0,
    residual_output_l2: float = 0.0,
) -> tuple[float, float, float, float]:
    total = torch.zeros(4)
    seen = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(output_scale.device) for key, value in batch.items()}
            mixed, one_step, rollout, residual = _mixed_loss(
                model,
                batch,
                output_scale,
                scales,
                one_step_weight,
                rollout_weight,
                residual_weight_l2,
                residual_output_l2,
            )
            total += torch.tensor(
                (float(mixed), float(one_step), float(rollout), float(residual))
            ) * len(batch["history"])
            seen += len(batch["history"])
    return tuple((total / seen).tolist())


def _train_variant(
    *,
    name: str,
    train: list[TCNFlight],
    validation: list[TCNFlight],
    test: list[TCNFlight],
    horizon: int,
    epochs: int,
    batch_size: int,
    stride: int,
    one_step_weight: float,
    rollout_weight: float,
    seed: int,
    device: torch.device,
    output: Path,
    residual: bool = False,
    residual_width: int = 8,
    residual_weight_l2: float = 0.1,
    residual_output_l2: float = 5.0,
) -> dict[str, object]:
    torch.manual_seed(seed)
    train_data = MixedWindows(train, horizon, stride)
    validation_data = MixedWindows(validation, horizon, stride)
    constructor = ResidualParityMotorNet if residual else ParityMotorNet
    extra = {"residual_width": residual_width} if residual else {}
    model = constructor(
        canonical_dim(include_voltage=True),
        output_dim(),
        hidden_per_type=24,
        **extra,
    ).to(device)
    model.reads_voltage = True
    _set_input_normalization(model, [_canonical_features(flight) for flight in train])
    output_rows = torch.cat([flight.labels for flight in train])
    output_scale = output_rows.std(0).clamp_min(1e-6).to(device)
    scales = _state_scales(train, device)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=_collate
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, collate_fn=_collate
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    records = []
    started = time.perf_counter()
    residual_params = (
        sum(parameter.numel() for parameter in model.residual.parameters())
        if residual
        else 0
    )
    print(
        f"{name:12s}  horizon {horizon} ({horizon * 0.0025:.3f}s)  "
        f"train {len(train_data)}  val {len(validation_data)}  "
        f"residual {residual_params}",
        flush=True,
    )
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            totals = torch.zeros(4)
            seen = 0
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                mixed, one_step, rollout, residual_loss = _mixed_loss(
                    model,
                    batch,
                    output_scale,
                    scales,
                    one_step_weight,
                    rollout_weight,
                    residual_weight_l2,
                    residual_output_l2,
                )
                optimizer.zero_grad(set_to_none=True)
                mixed.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                totals += torch.tensor(
                    (
                        float(mixed.detach()),
                        float(one_step.detach()),
                        float(rollout.detach()),
                        float(residual_loss.detach()),
                    )
                ) * len(batch["history"])
                seen += len(batch["history"])
            train_mixed, train_one, train_roll, train_residual = (totals / seen).tolist()
            scheduler.step()
        else:
            train_mixed, train_one, train_roll, train_residual = _mean_mixed(
                model,
                train_loader,
                output_scale,
                scales,
                one_step_weight,
                rollout_weight,
                residual_weight_l2,
                residual_output_l2,
            )
        val_mixed, val_one, val_roll, val_residual = _mean_mixed(
            model,
            validation_loader,
            output_scale,
            scales,
            one_step_weight,
            rollout_weight,
            residual_weight_l2,
            residual_output_l2,
        )
        saved = val_mixed < best_loss
        if saved:
            best_loss = val_mixed
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        records.append(
            {
                "epoch": epoch,
                "train_mixed": train_mixed,
                "train_one_step": train_one,
                "train_rollout": train_roll,
                "train_residual": train_residual,
                "val_mixed": val_mixed,
                "val_one_step": val_one,
                "val_rollout": val_roll,
                "val_residual": val_residual,
            }
        )
        print(
            f"{name:12s}  epoch {epoch:03d}  mixed {train_mixed:.6f}  "
            f"one {train_one:.6f}  roll {train_roll:.6f}  "
            f"res {train_residual:.4f}  val {val_mixed:.6f}"
            f"{(' saved' if saved else '')}",
            flush=True,
        )
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output / f"{name}.pt")
    try:
        from scripts.run_neurobem_parity_one_step import ParityWindows
    except ModuleNotFoundError:
        from run_neurobem_parity_one_step import ParityWindows
    one_step = _one_step_metrics(model, ParityWindows(test, 1), device, batch_size)
    rollout = _rollout_metrics(
        model,
        test,
        device,
        horizon=200,
        start_stride=20,
        batch_size=128,
    )
    residual_rms = 0.0
    if residual:
        energies = []
        model.eval()
        with torch.inference_mode():
            for batch in validation_loader:
                history = batch["history"].to(device)
                hidden = model.initial_hidden(len(history), history)
                for step in range(history.shape[1]):
                    _, hidden = model.step(history[:, step], hidden)
                    if model.last_residual is not None:
                        energies.append(model.last_residual.detach().square().mean().sqrt())
                if len(energies) >= 64:
                    break
        residual_rms = float(torch.stack(energies).mean()) if energies else 0.0
    return {
        "horizon_steps": horizon,
        "horizon_seconds": horizon * 0.0025,
        "residual": residual,
        "residual_parameters": residual_params,
        "residual_weight_l2": residual_weight_l2,
        "residual_output_l2": residual_output_l2,
        "residual_rms": residual_rms,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_mixed": best_loss,
        "training_seconds": time.perf_counter() - started,
        "history": records,
        "test_one_step": one_step,
        "test_rollout_mae": rollout,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("runs/neurobem_parity_mixed"))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--one-step-weight", type=float, default=1.0)
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--residual-width", type=int, default=8)
    parser.add_argument("--residual-weight-l2", type=float, default=0.1)
    parser.add_argument("--residual-output-l2", type=float, default=5.0)
    parser.add_argument(
        "--horizons",
        default="40,100,200",
        help="comma-separated 400 Hz horizons, e.g. 40,100,200",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train, validation, test = _load_split(args.data, args.seed)
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    records: dict[str, object] = {
        "config": {
            "objective": "one-step force/torque plus open-loop state MSE",
            "rate_hz": 400,
            "history_seconds": 0.05,
            "epochs": args.epochs,
            "stride": args.stride,
            "one_step_weight": args.one_step_weight,
            "rollout_weight": args.rollout_weight,
            "seed": args.seed,
            "device": str(device),
            "horizons": horizons,
            "residual": args.residual,
            "residual_width": args.residual_width,
            "residual_weight_l2": args.residual_weight_l2,
            "residual_output_l2": args.residual_output_l2,
        },
        "models": {},
    }
    for horizon in horizons:
        name = f"{'mix_res_h' if args.residual else 'mix_h'}{horizon}"
        records["models"][name] = _train_variant(
            name=name,
            train=train,
            validation=validation,
            test=test,
            horizon=horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
            stride=args.stride,
            one_step_weight=args.one_step_weight,
            rollout_weight=args.rollout_weight,
            seed=args.seed,
            device=device,
            output=args.output,
            residual=args.residual,
            residual_width=args.residual_width,
            residual_weight_l2=args.residual_weight_l2,
            residual_output_l2=args.residual_output_l2,
        )
        (args.output / "results.json").write_text(json.dumps(records, indent=2))
    print(f"results written to {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
