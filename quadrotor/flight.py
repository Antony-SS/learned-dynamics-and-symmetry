"""Shared flight-log tensors for a rigid-body drone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PreparedFlight:
    name: str
    position: np.ndarray
    rotation: np.ndarray
    velocity_world: np.ndarray
    omega_body: np.ndarray
    motors: np.ndarray  # rotor speeds, rad/s
    dt: np.ndarray
    attitude: str

    @property
    def n_poses(self) -> int:
        return int(self.position.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.motors.shape[0])
