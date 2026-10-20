"""Prepared state for the NanoBench system-identification rollout.

Each window is initialized once from Vicon. Thereafter the network propagates
position, attitude, linear velocity, and angular velocity without reading
another logged state. The EKF quaternion is only a fallback for a missing
initial Vicon attitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quadrotor.data import Flight, load_data_rich_split, load_split

DT = 0.01
HORIZON_STEPS = 50

ABSOLUTE_INPUTS = (
    "x", "y", "z",
    "vx_world", "vy_world", "vz_world",
    "rotation_x", "rotation_y", "rotation_z",
    "omega_x_body", "omega_y_body", "omega_z_body",
    "m1", "m2", "m3", "m4",
)
CANONICAL_INPUTS = (
    "gravity_x_body", "gravity_y_body", "gravity_z_body",
    "vx_body", "vy_body", "vz_body",
    "omega_x_body", "omega_y_body", "omega_z_body",
    "m1", "m2", "m3", "m4",
)
VOLTAGE_INPUT = ("battery_voltage",)
OUTPUTS = (
    "acceleration_x_body", "acceleration_y_body", "acceleration_z_body",
    "angular_acceleration_x_body", "angular_acceleration_y_body",
    "angular_acceleration_z_body",
)


def absolute_dim(include_voltage: bool = False) -> int:
    return len(ABSOLUTE_INPUTS) + int(include_voltage)


def canonical_dim(include_voltage: bool = False) -> int:
    return len(CANONICAL_INPUTS) + int(include_voltage)


def output_dim() -> int:
    return len(OUTPUTS)


@dataclass
class PreparedFlight:
    name: str
    position: np.ndarray
    rotation: np.ndarray
    velocity_world: np.ndarray
    omega_body: np.ndarray
    pwm: np.ndarray
    voltage: np.ndarray
    dt: np.ndarray
    attitude: str

    @property
    def n_poses(self) -> int:
        return int(self.position.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.pwm.shape[0])


def orientation(flight: Flight) -> tuple[np.ndarray, str]:
    """Rotation matrices and which sensor they came from."""
    if np.isfinite(flight.rotation).all():
        return flight.rotation, "vicon"
    if np.isfinite(flight.ekf_rotation).all():
        return flight.ekf_rotation, "ekf"
    raise ValueError(f"{flight.name} has no finite Vicon or EKF attitude")


def prepare(flight: Flight) -> PreparedFlight:
    rotation, source = orientation(flight)
    position = np.asarray(flight.position, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32)
    velocity_world = np.asarray(flight.velocity_world, dtype=np.float32)
    omega_world = np.asarray(flight.angular_velocity_world, dtype=np.float32)
    omega_body = np.einsum("nji,nj->ni", rotation, omega_world).astype(np.float32)
    pwm = np.asarray(flight.pwm[:-1], dtype=np.float32)
    voltage = np.asarray(flight.voltage[:-1], dtype=np.float32)
    dt = np.diff(flight.time).astype(np.float32)
    if pwm.shape[0] != dt.shape[0]:
        raise ValueError(f"{flight.name} motors and time steps disagree")
    if np.any(dt <= 0.0):
        raise ValueError(f"{flight.name} has a non-positive time step")
    return PreparedFlight(
        name=flight.name,
        position=position,
        rotation=rotation,
        velocity_world=velocity_world,
        omega_body=omega_body,
        pwm=pwm,
        voltage=voltage,
        dt=dt,
        attitude=source,
    )


def load_prepared(
    root: str | None = None,
    *,
    data_rich: bool = False,
    seed: int = 0,
) -> tuple[list[PreparedFlight], list[PreparedFlight], list[PreparedFlight]]:
    if data_rich:
        train, validation, test = load_data_rich_split(root, seed=seed)
    else:
        train, validation, test = load_split(root)
    return (
        [prepare(flight) for flight in train],
        [prepare(flight) for flight in validation],
        [prepare(flight) for flight in test],
    )
