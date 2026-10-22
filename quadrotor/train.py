"""Open-loop fit of a coherent recurrent quadrotor transition.

A window is seeded once from Vicon. The network then propagates its own
position, attitude, velocity, and angular velocity from the motor sequence.
It predicts only body-frame linear and angular acceleration; kinematics
determine every pose update. The MSE covers all four observed state components.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from learned_dynamics.train import TrainResult
from quadrotor.canonical import HORIZON_STEPS, PreparedFlight
from quadrotor.geometry import so3_log_batch
from quadrotor.rollout import held_out_rmse

_ANGLE_EPS = 1e-8


def module_device(model: torch.nn.Module) -> torch.device:
    """Device of a module that may have buffers but no parameters."""
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


def _hat(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zeros = torch.zeros_like(x)
    row0 = torch.stack((zeros, -z, y), dim=-1)
    row1 = torch.stack((z, zeros, -x), dim=-1)
    row2 = torch.stack((-y, x, zeros), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _so3_exp(phi: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True)
    small = theta < _ANGLE_EPS
    theta_safe = theta.clamp_min(_ANGLE_EPS)
    axis = phi / theta_safe
    skew = _hat(axis)
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(phi.shape[0], 3, 3)
    sine = torch.sin(theta_safe)[..., None]
    cosine = torch.cos(theta_safe)[..., None]
    rotation = eye + sine * skew + (1.0 - cosine) * (skew @ skew)
    return torch.where(small[..., None], eye + _hat(phi), rotation)


def _so3_log(rotation: torch.Tensor) -> torch.Tensor:
    cosine = ((torch.diagonal(rotation, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    half_vee = 0.5 * torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.linalg.vector_norm(half_vee, dim=-1)
    theta = torch.atan2(sine, cosine)
    scale = torch.where(sine < _ANGLE_EPS, torch.ones_like(sine), theta / sine)
    return scale.unsqueeze(-1) * half_vee


def _features(
    model: torch.nn.Module,
    position: torch.Tensor,
    rotation: torch.Tensor,
    velocity_world: torch.Tensor,
    omega_body: torch.Tensor,
    pwm: torch.Tensor,
    voltage: torch.Tensor | None = None,
) -> torch.Tensor:
    if getattr(model, "reads_pose", False):
        parts = (position, velocity_world, _so3_log(rotation), omega_body, pwm)
        features = torch.cat(parts, dim=-1)
        if getattr(model, "reads_voltage", False):
            if voltage is None:
                raise ValueError("voltage-aware model requires battery voltage")
            features = torch.cat((features, voltage.unsqueeze(-1)), dim=-1)
        if getattr(model, "reads_height", False):
            features = torch.cat((features, position[..., 2:3]), dim=-1)
        return features
    velocity_body = torch.bmm(rotation.transpose(1, 2), velocity_world.unsqueeze(-1)).squeeze(-1)
    up_world = rotation.new_tensor((0.0, 0.0, 1.0)).expand(rotation.shape[0], 3)
    gravity_body = torch.bmm(
        rotation.transpose(1, 2), up_world.unsqueeze(-1)
    ).squeeze(-1)
    features = torch.cat((gravity_body, velocity_body, omega_body, pwm), dim=-1)
    if getattr(model, "reads_voltage", False):
        if voltage is None:
            raise ValueError("voltage-aware model requires battery voltage")
        features = torch.cat((features, voltage.unsqueeze(-1)), dim=-1)
    if getattr(model, "reads_height", False):
        features = torch.cat((features, position[..., 2:3]), dim=-1)
    return features


def _apply(
    position: torch.Tensor,
    rotation: torch.Tensor,
    velocity_world: torch.Tensor,
    omega_body: torch.Tensor,
    acceleration: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dt = dt.unsqueeze(-1)
    acceleration_world = torch.bmm(
        rotation, acceleration[:, 0:3].unsqueeze(-1)
    ).squeeze(-1)
    angular_acceleration = acceleration[:, 3:6]
    position = position + velocity_world * dt + 0.5 * acceleration_world * dt.square()
    velocity_world = velocity_world + acceleration_world * dt
    rotation_increment = (omega_body + 0.5 * angular_acceleration * dt) * dt
    rotation = torch.bmm(rotation, _so3_exp(rotation_increment))
    omega_body = omega_body + angular_acceleration * dt
    return position, rotation, velocity_world, omega_body


def _starts(flights: list[PreparedFlight], horizon: int, stride: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for index, flight in enumerate(flights):
        last = flight.n_steps - horizon
        if last < 0:
            raise ValueError(
                f"{flight.name} has {flight.n_steps} steps, shorter than a {horizon} step window"
            )
        for start in range(0, last + 1, stride):
            pairs.append((index, start))
    if not pairs:
        raise ValueError("no training windows")
    return pairs


def _integrate(
    model: torch.nn.Module,
    pwm: torch.Tensor,
    dt: torch.Tensor,
    position: torch.Tensor,
    rotation: torch.Tensor,
    velocity_world: torch.Tensor,
    omega_body: torch.Tensor,
    voltage: torch.Tensor | None = None,
    hidden: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll the complete predicted state through the motor sequence."""
    positions, rotations, velocities, omegas = [], [], [], []
    if hidden is None:
        hidden = model.initial_hidden(pwm.shape[0], pwm)
    advance = getattr(model, "advance", None)
    for index in range(pwm.shape[1]):
        if callable(advance):
            position, rotation, velocity_world, omega_body, hidden = advance(
                position,
                rotation,
                velocity_world,
                omega_body,
                pwm[:, index],
                dt[:, index],
                hidden,
            )
        else:
            features = _features(
                model,
                position,
                rotation,
                velocity_world,
                omega_body,
                pwm[:, index],
                None if voltage is None else voltage[:, index],
            )
            acceleration, hidden = model.step(features, hidden)
            position, rotation, velocity_world, omega_body = _apply(
                position, rotation, velocity_world, omega_body, acceleration, dt[:, index]
            )
        positions.append(position)
        rotations.append(rotation)
        velocities.append(velocity_world)
        omegas.append(omega_body)
    return (
        torch.stack(positions, dim=1),
        torch.stack(rotations, dim=1),
        torch.stack(velocities, dim=1),
        torch.stack(omegas, dim=1),
    )


def prime_hidden(
    model: torch.nn.Module,
    flights: list[PreparedFlight],
    pairs: list[tuple[int, int]],
    warm_steps: int,
    voltage_value: float | None = None,
) -> torch.Tensor | None:
    """Teacher-force ``warm_steps`` Vicon frames ending at each window seed.

    This is the NeuroBEM-style prime (20 samples = 50 ms at 400 Hz), not a
    walk over the whole flight.
    """
    if warm_steps <= 0:
        return None
    device = module_device(model)
    reference = torch.zeros(len(pairs), 1, device=device)
    hidden = model.initial_hidden(len(pairs), reference)
    if hidden.numel() == 0:
        return hidden
    for offset in range(warm_steps):
        position, rotation, velocity, omega, pwm, voltage = [], [], [], [], [], []
        for index, start in pairs:
            flight = flights[index]
            step = min(max(0, start - warm_steps + offset), flight.n_steps - 1)
            position.append(flight.position[step])
            rotation.append(flight.rotation[step])
            velocity.append(flight.velocity_world[step])
            omega.append(flight.omega_body[step])
            pwm.append(flight.pwm[step])
            voltage.append(flight.voltage[step])
        voltage_t = torch.as_tensor(np.stack(voltage), device=device)
        if voltage_value is not None:
            voltage_t = torch.full_like(voltage_t, float(voltage_value))
        features = _features(
            model,
            torch.as_tensor(np.stack(position), device=device),
            torch.as_tensor(np.stack(rotation), device=device),
            torch.as_tensor(np.stack(velocity), device=device),
            torch.as_tensor(np.stack(omega), device=device),
            torch.as_tensor(np.stack(pwm), device=device),
            voltage_t,
        )
        step_fn = getattr(model, "step", None)
        if callable(step_fn):
            _acceleration, hidden = step_fn(features, hidden)
        else:
            neural = getattr(model, "neural", None)
            if neural is not None and callable(getattr(neural, "step", None)):
                _acceleration, hidden = neural.step(features, hidden)
    return hidden


def _batch_mse(
    model: torch.nn.Module,
    flights: list[PreparedFlight],
    pairs: list[tuple[int, int]],
    horizon: int,
    weights: torch.Tensor,
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    hidden_cache: list[list[torch.Tensor]] | None = None,
    warm_steps: int = 0,
) -> torch.Tensor:
    pwm = []
    voltage = []
    dt = []
    position = []
    rotation = []
    velocity = []
    omega = []
    for index, start in pairs:
        flight = flights[index]
        stop = start + horizon
        pwm.append(flight.pwm[start:stop])
        voltage.append(flight.voltage[start:stop])
        dt.append(flight.dt[start:stop])
        position.append(flight.position[start : stop + 1])
        rotation.append(flight.rotation[start : stop + 1])
        velocity.append(flight.velocity_world[start : stop + 1])
        omega.append(flight.omega_body[start : stop + 1])
    device = module_device(model)
    if hasattr(model, "reset_residual_stats"):
        model.reset_residual_stats()
    pwm_t = torch.as_tensor(np.stack(pwm), device=device)
    voltage_t = torch.as_tensor(np.stack(voltage), device=device)
    dt_t = torch.as_tensor(np.stack(dt), device=device)
    position_t = torch.as_tensor(np.stack(position), device=device)
    rotation_t = torch.as_tensor(np.stack(rotation), device=device)
    velocity_t = torch.as_tensor(np.stack(velocity), device=device)
    omega_t = torch.as_tensor(np.stack(omega), device=device)
    hidden = None
    if hidden_cache is not None:
        hidden = torch.cat(
            [hidden_cache[index][start].to(device) for index, start in pairs],
            dim=0,
        )
    elif warm_steps > 0:
        hidden = prime_hidden(model, flights, pairs, warm_steps)
    position_hat, rotation_hat, velocity_hat, omega_hat = _integrate(
        model,
        pwm_t,
        dt_t,
        position_t[:, 0],
        rotation_t[:, 0],
        velocity_t[:, 0],
        omega_t[:, 0],
        voltage_t,
        hidden=hidden,
    )
    relative = torch.matmul(rotation_t[:, 1:].transpose(-1, -2), rotation_hat)
    errors = (
        (position_hat - position_t[:, 1:]) / scales[0],
        _so3_log(relative) / scales[1],
        (velocity_hat - velocity_t[:, 1:]) / scales[2],
        (omega_hat - omega_t[:, 1:]) / scales[3],
    )
    per_step = torch.cat([error.pow(2) for error in errors], dim=-1)
    loss = (per_step * weights).mean()
    if (
        hasattr(model, "residual_penalty")
        and getattr(model, "_residual_count", 0)
        and getattr(model, "_residual_energy", None) is not None
    ):
        energy = model._residual_energy / model._residual_count
        loss = loss + model.residual_penalty(
            energy,
            float(getattr(model, "residual_weight_l2", 0.1)),
            float(getattr(model, "residual_output_l2", 5.0)),
        )
    return loss


def _mean_mse(
    model: torch.nn.Module,
    flights: list[PreparedFlight],
    pairs: list[tuple[int, int]],
    horizon: int,
    batch_size: int,
    weights: torch.Tensor,
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    hidden_cache: list[list[torch.Tensor]] | None = None,
    warm_steps: int = 0,
) -> torch.Tensor:
    total = torch.zeros(())
    seen = 0
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        total = total + _batch_mse(
            model, flights, chunk, horizon, weights, scales, hidden_cache, warm_steps
        ).detach() * len(chunk)
        seen += len(chunk)
        if start > 0 and start % (batch_size * 20) == 0:
            print(f"           scored {seen}/{len(pairs)} windows", flush=True)
    return total / seen


def _state_scales(
    flights: list[PreparedFlight], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One scalar per geometric state block, fit on training data only."""
    position = np.concatenate([flight.position for flight in flights])
    velocity = np.concatenate([flight.velocity_world for flight in flights])
    omega = np.concatenate([flight.omega_body for flight in flights])
    rotation = np.concatenate([flight.rotation for flight in flights])
    rotation_vector = _so3_log(torch.as_tensor(rotation, dtype=torch.float32))
    values = (
        float(np.std(position)),
        float(rotation_vector.std()),
        float(np.std(velocity)),
        float(np.std(omega)),
    )
    return tuple(
        torch.tensor(max(value, 1e-6), device=device).view(1, 1, 1)
        for value in values
    )


def _input_rows(model: torch.nn.Module, flights: list[PreparedFlight]) -> np.ndarray:
    """Ground-truth versions of the state inputs, from training flights only."""
    rows = []
    for flight in flights:
        if getattr(model, "reads_pose", False):
            rotation_vector = so3_log_batch(flight.rotation[:-1])
            rows.append(
                np.concatenate(
                    (
                        flight.position[:-1],
                        flight.velocity_world[:-1],
                        rotation_vector,
                        flight.omega_body[:-1],
                        flight.pwm,
                        *(
                            (flight.voltage[:, None],)
                            if getattr(model, "reads_voltage", False)
                            else ()
                        ),
                        *(
                            (flight.position[:-1, 2:3],)
                            if getattr(model, "reads_height", False)
                            else ()
                        ),
                    ),
                    axis=1,
                )
            )
        else:
            up_world = np.broadcast_to(
                np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
                (flight.n_steps, 3),
            )
            gravity_body = np.einsum(
                "nji,nj->ni", flight.rotation[:-1], up_world
            )
            velocity_body = np.einsum(
                "nji,nj->ni", flight.rotation[:-1], flight.velocity_world[:-1]
            )
            rows.append(
                np.concatenate(
                    (
                        gravity_body,
                        velocity_body,
                        flight.omega_body[:-1],
                        flight.pwm,
                        *(
                            (flight.voltage[:, None],)
                            if getattr(model, "reads_voltage", False)
                            else ()
                        ),
                        *(
                            (flight.position[:-1, 2:3],)
                            if getattr(model, "reads_height", False)
                            else ()
                        ),
                    ),
                    axis=1,
                )
            )
    return np.concatenate(rows, axis=0).astype(np.float32)


def _configure_input_normalization(
    model: torch.nn.Module, flights: list[PreparedFlight]
) -> None:
    """Fit input normalization on the training split without breaking symmetry."""
    rows = _input_rows(model, flights)
    center = rows.mean(axis=0)
    scale = rows.std(axis=0)
    if getattr(model, "longitudinal_mirror", False):
        # Left-right odd channels only. Do not share front-back scales.
        center[1] = 0.0
        center[4] = 0.0
        center[6] = 0.0
        center[8] = 0.0
        center[9:13] = float(np.mean(rows[:, 9:13]))
        scale[9:13] = float(np.std(rows[:, 9:13]))
    elif getattr(model, "mirror", False):
        # Every sign-changing channel is centered at zero. Shared horizontal
        # scales preserve both mirrors and a possible future 90-degree rotation.
        center[0:2] = 0.0
        center[3:9] = 0.0
        scale[0:2] = float(np.sqrt(np.mean(rows[:, 0:2] ** 2)))
        scale[3:5] = float(np.sqrt(np.mean(rows[:, 3:5] ** 2)))
        scale[6:8] = float(np.sqrt(np.mean(rows[:, 6:8] ** 2)))
        scale[8] = float(np.sqrt(np.mean(rows[:, 8] ** 2)))
        motor_center = float(np.mean(rows[:, 9:13]))
        motor_scale = float(np.std(rows[:, 9:13]))
        center[9:13] = motor_center
        scale[9:13] = motor_scale
    model.set_input_normalization(
        torch.as_tensor(center, dtype=torch.float32),
        torch.as_tensor(scale, dtype=torch.float32),
    )


def train_open_loop(
    model: torch.nn.Module,
    train_flights: list[PreparedFlight],
    val_flights: list[PreparedFlight],
    test_flights: list[PreparedFlight],
    *,
    epochs: int = 500,
    lr: float = 1e-3,
    rollout_every: int = 10,
    grad_clip: float = 0.5,
    batch_size: int = 256,
    stride: int = 1,
    horizon: int = HORIZON_STEPS,
    horizon_weight: float = 0.0,
    omega_end_weight: float = 1.0,
    score_full_test: bool = False,
    name: str = "quadrotor",
    extra_metrics: Callable[[torch.nn.Module, list[PreparedFlight]], dict[str, float]]
    | None = None,
    warm_start: bool = False,
    warm_steps: int = 0,
) -> TrainResult:
    """Fit open-loop windows and restore the best validation path.

    The checkpoint is the standardized full-state MSE on the multisine
    validation tails.     ``horizon_weight=0`` gives every rollout step equal
    weight. ``omega_end_weight`` ramps the ω term from 1 to that value
    along the window; 1 leaves the official loss unchanged. Category B
    is never used to choose the epoch.
    """
    if epochs < 0:
        raise ValueError(f"epochs must be non-negative, got {epochs}")
    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if rollout_every <= 0:
        raise ValueError(f"rollout_every must be positive, got {rollout_every}")
    if horizon < 50:
        raise ValueError(f"horizon must be at least 50 steps to report h=10 and h=50, got {horizon}")
    if omega_end_weight <= 0:
        raise ValueError(f"omega_end_weight must be positive, got {omega_end_weight}")
    probe = torch.zeros(
        1, model.input_center.numel(), device=module_device(model)
    )
    if model(probe).shape[-1] != 6:
        raise ValueError("model must write 6 body-frame accelerations")
    train_pairs = _starts(train_flights, horizon, stride)
    val_pairs = _starts(val_flights, horizon, stride)
    print(
        f"{name:10s}  {horizon} steps  stride {stride}  "
        f"train {len(train_pairs)}  val {len(val_pairs)}  "
        f"test flights {len(test_flights)}  batch {batch_size}  "
        f"weight {'uniform' if horizon_weight == 0 else f'exp(-{horizon_weight:g} h)'}  "
        f"omega {'flat' if omega_end_weight == 1 else f'ramp 1→{omega_end_weight:g}'}  "
        f"gru {'warm ' + str(warm_steps) + ' steps' if warm_steps > 0 else ('warm' if warm_start else 'cold')}",
        flush=True,
    )

    _configure_input_normalization(model, train_flights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    device = module_device(model)
    time_weight = torch.exp(-horizon_weight * torch.arange(horizon, device=device))
    time_weight = (time_weight / time_weight.mean()).view(1, horizon, 1)
    channel = torch.ones(1, horizon, 4, device=device)
    if omega_end_weight != 1.0:
        # Late steps: pose is H² and eats α. Ramp ω so the far rate stays loud.
        ramp = torch.arange(1, horizon + 1, device=device, dtype=torch.float32) / horizon
        channel[0, :, 3] = 1.0 + (float(omega_end_weight) - 1.0) * ramp
    weights = time_weight * channel.repeat_interleave(3, dim=-1)
    scales = _state_scales(train_flights, device)
    history: list[dict] = []
    best_error = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    order = torch.arange(len(train_pairs))

    def _hidden_cache(flights: list[PreparedFlight]) -> list[list[torch.Tensor]] | None:
        if warm_steps > 0 or not warm_start:
            return None
        from quadrotor.evaluate import teacher_forced_hiddens

        print(f"{name:10s}  caching full-flight warm hiddens ({len(flights)} flights)...", flush=True)
        model.eval()
        with torch.no_grad():
            return [teacher_forced_hiddens(model, flight) for flight in flights]

    for epoch in range(epochs + 1):
        print(f"{name:10s}  epoch {epoch:03d} starting", flush=True)
        train_cache = _hidden_cache(train_flights)
        if epoch == 0:
            model.eval()
            with torch.no_grad():
                loss = _mean_mse(
                    model,
                    train_flights,
                    train_pairs,
                    horizon,
                    batch_size,
                    weights,
                    scales,
                    train_cache,
                    warm_steps,
                )
        else:
            model.train()
            order = order[torch.randperm(len(order))]
            total = torch.zeros(())
            seen = 0
            for start in range(0, len(order), batch_size):
                chosen = [train_pairs[int(i)] for i in order[start : start + batch_size]]
                batch_loss = _batch_mse(
                    model,
                    train_flights,
                    chosen,
                    horizon,
                    weights,
                    scales,
                    train_cache,
                    warm_steps,
                )
                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                total = total + batch_loss.detach() * len(chosen)
                seen += len(chosen)
            loss = total / seen
            scheduler.step()

        val_cache = _hidden_cache(val_flights)
        with torch.no_grad():
            model.eval()
            val_loss = _mean_mse(
                model,
                val_flights,
                val_pairs,
                horizon,
                batch_size,
                weights,
                scales,
                val_cache,
                warm_steps,
            )
        row = {
            "epoch": epoch,
            "train_loss": float(loss.detach()),
            "val_loss": float(val_loss.detach()),
        }
        if extra_metrics is not None:
            row.update(extra_metrics(model, val_flights))
        mark = ""
        if row["val_loss"] < best_error:
            best_error = row["val_loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            mark = "  saved"
        if epoch % rollout_every == 0 or epoch == epochs:
            # Imported here to avoid the module cycle: evaluate reuses this
            # file's differentiable state integrator.
            from quadrotor.evaluate import benchmark_mae

            validation_mae = benchmark_mae(
                model,
                val_flights,
                horizon=horizon,
                batch_size=batch_size,
                warm_start=warm_start,
                warm_steps=warm_steps,
            )
            for state, values in validation_mae.items():
                row[f"val_{state}_mae_h10"] = float(values[9])
                row[f"val_{state}_mae_h50"] = float(values[49])
                if horizon >= 100:
                    row[f"val_{state}_mae_h100"] = float(values[99])
                    row[f"val_{state}_mae_h{horizon}"] = float(values[-1])
            rmse_note = (
                f"  h10 MAE p {validation_mae['position'][9]:.4f} m"
                f" v {validation_mae['velocity'][9]:.4f} m/s"
                f" R {validation_mae['rotation'][9]:.4f} rad"
                f" w {validation_mae['omega'][9]:.4f} rad/s"
                f"  h50 MAE p {validation_mae['position'][49]:.4f} m"
                f" v {validation_mae['velocity'][49]:.4f} m/s"
                f" R {validation_mae['rotation'][49]:.4f} rad"
                f" w {validation_mae['omega'][49]:.4f} rad/s{mark}"
            )
            if horizon >= 100:
                rmse_note += (
                    f"  h100 p {validation_mae['position'][99]:.4f} m"
                    f" R {validation_mae['rotation'][99]:.4f} rad"
                    f" w {validation_mae['omega'][99]:.4f} rad/s"
                    f"  h{horizon} p {validation_mae['position'][-1]:.4f} m"
                    f" R {validation_mae['rotation'][-1]:.4f} rad"
                    f" w {validation_mae['omega'][-1]:.4f} rad/s"
                )
        else:
            for state in ("position", "velocity", "rotation", "omega"):
                row[f"val_{state}_mae_h10"] = float("nan")
                row[f"val_{state}_mae_h50"] = float("nan")
                if horizon >= 100:
                    row[f"val_{state}_mae_h100"] = float("nan")
                    row[f"val_{state}_mae_h{horizon}"] = float("nan")
            rmse_note = mark
        print(
            f"{name:10s}  epoch {epoch:03d}  weighted mse {row['train_loss']:.6e}  "
            f"val {row['val_loss']:.6e}{rmse_note}",
            flush=True,
        )
        history.append(row)
        if epoch == epochs:
            break

    model.load_state_dict(best_state)
    summary = (
        f"{name:10s}  best epoch {best_epoch:03d}  "
        f"val state MSE {best_error:.6e}"
    )
    if score_full_test:
        test_error = float(held_out_rmse(model, test_flights))
        summary += f"  full-test RMSE {test_error:.4f} m"
    print(summary)
    return TrainResult(history=history, best_epoch=best_epoch, best_error=best_error)
