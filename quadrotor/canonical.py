"""Prepared flight tensors shared by the Crazyflie loaders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT = 0.01
HORIZON_STEPS = 50

CANONICAL_INPUTS = (
    "gravity_x_body", "gravity_y_body", "gravity_z_body",
    "vx_body", "vy_body", "vz_body",
    "omega_x_body", "omega_y_body", "omega_z_body",
    "m1", "m2", "m3", "m4",
)
OUTPUTS = (
    "acceleration_x_body", "acceleration_y_body", "acceleration_z_body",
    "angular_acceleration_x_body", "angular_acceleration_y_body",
    "angular_acceleration_z_body",
)


def canonical_dim() -> int:
    return len(CANONICAL_INPUTS)


def output_dim() -> int:
    return len(OUTPUTS)


@dataclass
class PreparedFlight:
    name: str
    position: np.ndarray
    rotation: np.ndarray
    velocity_world: np.ndarray
    omega_body: np.ndarray
    motors: np.ndarray  # IDSIA: rad/s. NeuroBEM: logged motor speeds (RPM).
    dt: np.ndarray
    attitude: str

    @property
    def n_poses(self) -> int:
        return int(self.position.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.motors.shape[0])
