"""Full-flight open-loop integration of recurrent acceleration dynamics.

The first state comes from Vicon. Every later position, attitude, velocity,
and angular velocity is the model's own. The log supplies only motor commands.
"""

from __future__ import annotations

import numpy as np
import torch

from quadrotor.canonical import PreparedFlight


def open_loop(model: torch.nn.Module, flight: PreparedFlight) -> tuple[np.ndarray, float]:
    """Integrate from the first pose. Returns positions and 3D RMSE in meters."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            positions = _integrate(model, flight)
    finally:
        if was_training:
            model.train()
    error = positions - flight.position
    rmse = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
    return positions, rmse


def _integrate(model: torch.nn.Module, flight: PreparedFlight) -> np.ndarray:
    # Local import avoids the train -> rollout dependency during module import.
    from quadrotor.train import _integrate as integrate_state
    from quadrotor.train import module_device

    device = module_device(model)
    pwm = torch.as_tensor(flight.pwm, device=device).unsqueeze(0)
    voltage = torch.as_tensor(flight.voltage, device=device).unsqueeze(0)
    dt = torch.as_tensor(flight.dt, device=device).unsqueeze(0)
    position = torch.as_tensor(flight.position[0], device=device).unsqueeze(0)
    rotation = torch.as_tensor(flight.rotation[0], device=device).unsqueeze(0)
    velocity = torch.as_tensor(flight.velocity_world[0], device=device).unsqueeze(0)
    omega = torch.as_tensor(flight.omega_body[0], device=device).unsqueeze(0)
    predicted, _rotation, _velocity, _omega = integrate_state(
        model, pwm, dt, position, rotation, velocity, omega, voltage
    )
    return np.concatenate(
        (flight.position[0:1], predicted[0].detach().cpu().numpy()), axis=0
    )


def held_out_rmse(model: torch.nn.Module, flights: list[PreparedFlight]) -> float:
    """Length-weighted position RMSE across held-out flights."""
    total = 0.0
    weight = 0.0
    for flight in flights:
        _positions, rmse = open_loop(model, flight)
        total += (rmse**2) * flight.n_poses
        weight += flight.n_poses
    return float(np.sqrt(total / weight))
