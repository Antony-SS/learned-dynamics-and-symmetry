"""Shared flight-log tensors for a rigid-body drone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PreparedFlight:
    """One flight: a state sample at each pose from a ground truth drone flight trajectory, and the input held between poses.

    State fields share the sample axis N. motors and dt are the steps between
    them, length N - 1.

    Attributes:
        position: World position, shape (N, 3), meters.
        rotation: Body-to-world rotation, shape (N, 3, 3).
        velocity_world: World velocity, shape (N, 3), m/s.
        omega_body: Body rate, shape (N, 3), rad/s.
        motors: Rotor speeds over each step, shape (N - 1, 4), rad/s.
        dt: Step duration, shape (N - 1,), seconds.
    """
    name: str
    position: np.ndarray
    rotation: np.ndarray
    velocity_world: np.ndarray
    omega_body: np.ndarray
    motors: np.ndarray  # rotor speeds, rad/s
    dt: np.ndarray

    @property
    def n_poses(self) -> int:
        return int(self.position.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.motors.shape[0])
