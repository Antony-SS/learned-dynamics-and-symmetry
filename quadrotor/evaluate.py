"""NanoBench Task 1 metrics at horizons 1, 10, and 50."""

from __future__ import annotations

import numpy as np
import torch

from quadrotor.canonical import HORIZON_STEPS, PreparedFlight
from quadrotor.train import _integrate, _so3_log, module_device


def benchmark_mae(
    model: torch.nn.Module,
    flights: list[PreparedFlight],
    *,
    horizon: int = HORIZON_STEPS,
    batch_size: int = 256,
    start_stride: int = 1,
    start_offset: int = 0,
    voltage_value: float | None = None,
) -> dict[str, np.ndarray]:
    """Mean vector-norm error at every horizon, using every valid window."""
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
    try:
        with torch.no_grad():
            for offset in range(0, len(starts), batch_size):
                batch = starts[offset : offset + batch_size]
                pwm, voltage, dt, position, rotation, velocity, omega = (
                    [], [], [], [], [], [], []
                )
                for flight_index, start in batch:
                    flight = flights[flight_index]
                    stop = start + horizon
                    pwm.append(flight.pwm[start:stop])
                    voltage.append(flight.voltage[start:stop])
                    dt.append(flight.dt[start:stop])
                    position.append(flight.position[start : stop + 1])
                    rotation.append(flight.rotation[start : stop + 1])
                    velocity.append(flight.velocity_world[start : stop + 1])
                    omega.append(flight.omega_body[start : stop + 1])
                pwm_t = torch.as_tensor(np.stack(pwm), device=device)
                voltage_t = torch.as_tensor(np.stack(voltage), device=device)
                if voltage_value is not None:
                    voltage_t = torch.full_like(voltage_t, float(voltage_value))
                dt_t = torch.as_tensor(np.stack(dt), device=device)
                position_t = torch.as_tensor(np.stack(position), device=device)
                rotation_t = torch.as_tensor(np.stack(rotation), device=device)
                velocity_t = torch.as_tensor(np.stack(velocity), device=device)
                omega_t = torch.as_tensor(np.stack(omega), device=device)
                position_hat, rotation_hat, velocity_hat, omega_hat = _integrate(
                    model,
                    pwm_t,
                    dt_t,
                    position_t[:, 0],
                    rotation_t[:, 0],
                    velocity_t[:, 0],
                    omega_t[:, 0],
                    voltage_t,
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
