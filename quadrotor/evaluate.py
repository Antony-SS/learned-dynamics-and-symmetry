"""Open-loop MAE at every horizon for a kinematics-integrating model."""

from __future__ import annotations

import numpy as np
import torch

from quadrotor.canonical import HORIZON_STEPS, PreparedFlight
from quadrotor.geometry import _features, _integrate, _so3_log, module_device, prime_hidden


def _recurrent_step(
    model: torch.nn.Module,
    features: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    step = getattr(model, "step", None)
    if callable(step):
        _acceleration, hidden = step(features, hidden)
        return hidden
    neural = getattr(model, "neural", None)
    if neural is not None and callable(getattr(neural, "step", None)):
        _acceleration, hidden = neural.step(features, hidden)
        return hidden
    return hidden


def teacher_forced_hiddens(
    model: torch.nn.Module,
    flight: PreparedFlight,
) -> list[torch.Tensor]:
    """Hidden state at each step after seeing Vicon history up to that step."""
    device = module_device(model)
    hidden = model.initial_hidden(1, torch.zeros(1, 1, device=device))
    hiddens = []
    position = torch.as_tensor(flight.position, device=device)
    rotation = torch.as_tensor(flight.rotation, device=device)
    velocity = torch.as_tensor(flight.velocity_world, device=device)
    omega = torch.as_tensor(flight.omega_body, device=device)
    motors = torch.as_tensor(flight.motors, device=device)
    stateless = hidden.numel() == 0
    for start in range(flight.n_steps):
        hiddens.append(hidden if stateless else hidden.detach())
        if stateless:
            continue
        features = _features(
            model,
            position[start : start + 1],
            rotation[start : start + 1],
            velocity[start : start + 1],
            omega[start : start + 1],
            motors[start : start + 1],
        )
        hidden = _recurrent_step(model, features, hidden)
    return hiddens


def benchmark_mae(
    model: torch.nn.Module,
    flights: list[PreparedFlight],
    *,
    horizon: int = HORIZON_STEPS,
    batch_size: int = 256,
    start_stride: int = 1,
    start_offset: int = 0,
    warm_start: bool = False,
    warm_steps: int = 0,
) -> dict[str, np.ndarray]:
    """Mean vector-norm error at every horizon, using every valid window.

    ``warm_steps`` teacher-forces that many Vicon frames before the seed
    (NeuroBEM: 20). ``warm_start`` walks the whole flight; do not use that
    on 400 Hz logs.
    """
    if start_stride < 1:
        raise ValueError("start_stride must be positive")
    starts = [
        (flight_index, start)
        for flight_index, flight in enumerate(flights)
        for start in range(
            start_offset,
            flight.n_steps - horizon + 1,
            start_stride,
        )
    ]
    if not starts:
        raise ValueError(f"no flight is long enough for {horizon} steps")
    device = module_device(model)
    total = {
        name: torch.zeros(horizon, dtype=torch.float64)
        for name in ("position", "velocity", "rotation", "omega")
    }
    seen = 0
    was_training = model.training
    model.eval()
    warmed: dict[int, list[torch.Tensor]] = {}
    try:
        with torch.no_grad():
            if warm_start and warm_steps <= 0:
                for flight_index, flight in enumerate(flights):
                    warmed[flight_index] = teacher_forced_hiddens(model, flight)
            for offset in range(0, len(starts), batch_size):
                batch = starts[offset : offset + batch_size]
                motors, dt, position, rotation, velocity, omega = ([], [], [], [], [], [])
                hidden_batch = []
                for flight_index, start in batch:
                    flight = flights[flight_index]
                    stop = start + horizon
                    motors.append(flight.motors[start:stop])
                    dt.append(flight.dt[start:stop])
                    position.append(flight.position[start : stop + 1])
                    rotation.append(flight.rotation[start : stop + 1])
                    velocity.append(flight.velocity_world[start : stop + 1])
                    omega.append(flight.omega_body[start : stop + 1])
                    if warm_start and warm_steps <= 0:
                        hidden_batch.append(warmed[flight_index][start])
                motors_t = torch.as_tensor(np.stack(motors), device=device)
                dt_t = torch.as_tensor(np.stack(dt), device=device)
                position_t = torch.as_tensor(np.stack(position), device=device)
                rotation_t = torch.as_tensor(np.stack(rotation), device=device)
                velocity_t = torch.as_tensor(np.stack(velocity), device=device)
                omega_t = torch.as_tensor(np.stack(omega), device=device)
                hidden = torch.cat(hidden_batch, dim=0) if hidden_batch else None
                if warm_steps > 0:
                    hidden = prime_hidden(model, flights, batch, warm_steps)
                position_hat, rotation_hat, velocity_hat, omega_hat = _integrate(
                    model,
                    motors_t,
                    dt_t,
                    position_t[:, 0],
                    rotation_t[:, 0],
                    velocity_t[:, 0],
                    omega_t[:, 0],
                    hidden=hidden,
                )
                relative = torch.matmul(rotation_t[:, 1:].transpose(-1, -2), rotation_hat)
                errors = {
                    "position": torch.linalg.vector_norm(
                        position_hat - position_t[:, 1:], dim=-1
                    ),
                    "velocity": torch.linalg.vector_norm(
                        velocity_hat - velocity_t[:, 1:], dim=-1
                    ),
                    "rotation": torch.linalg.vector_norm(_so3_log(relative), dim=-1),
                    "omega": torch.linalg.vector_norm(
                        omega_hat - omega_t[:, 1:], dim=-1
                    ),
                }
                for name, error in errors.items():
                    total[name] += error.sum(dim=0).detach().cpu().to(torch.float64)
                seen += len(batch)
    finally:
        if was_training:
            model.train()
    return {name: (value / seen).numpy() for name, value in total.items()}
