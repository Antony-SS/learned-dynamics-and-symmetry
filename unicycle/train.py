"""Short open-loop fit of the unicycle window.

Each training sample is the center of a 5-second rollout. The loss is the xy
error of that rollout, integrated from the true pose at the start of the
window. Pass ``yaw_length`` to add wrapped heading in meters on that same
scale. An epoch is a shuffle of those windows in mini-batches. With heading
supervision, the same yaw-inclusive validation loss selects the checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from learned_dynamics.train import TrainResult
from unicycle.data import MAX_LINEAR_VELOCITY, ROBOT_DIAMETER, Trajectory, kept_steps
from unicycle.geometry import body_steps
from unicycle.models import history_slice, rollout_command_velocity
from unicycle.rollout import (
    held_out_heading_rmse,
    held_out_rmse,
    pooled_rollout_position_mse,
)

# One folder, overwritten on each training run. Plot cells read these files.
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "runs" / "unicycle"

HORIZON_S = 5.0
VAL_HORIZON_S = 30.0
VAL_STRIDE_S = 1.0
# Wrapped heading enters the loss as meters of arc after one second at the
# linear speed cap. On the unicycle's 5 s windows that matches the RMS of one
# position component, so yaw does not drown xy and xy does not drown yaw.
HEADING_LENGTH = MAX_LINEAR_VELOCITY
# Encoder velocity enters the 5 s loss as m/s and as yaw arc at half the
# robot diameter, so it sits on the same scale as a one-step pose error.
VELOCITY_WEIGHT = 1.0
_ANGLE_EPS = 1e-8


def _module_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return next(model.buffers()).device


def _se2_exp(xi: torch.Tensor) -> torch.Tensor:
    """Relative pose ``(x, y, φ)`` from tangent coordinates ``(ρx, ρy, φ)``."""
    phi = xi[..., 2]
    small = phi.abs() <= _ANGLE_EPS
    phi_safe = torch.where(small, torch.ones_like(phi), phi)
    a = torch.where(small, torch.ones_like(phi), torch.sin(phi_safe) / phi_safe)
    b = torch.where(small, 0.5 * phi, (1.0 - torch.cos(phi_safe)) / phi_safe)
    x = a * xi[..., 0] - b * xi[..., 1]
    y = b * xi[..., 0] + a * xi[..., 1]
    return torch.stack((x, y, phi), dim=-1)


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def _compose_relative(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(first[..., 2])
    sine = torch.sin(first[..., 2])
    x = first[..., 0] + cosine * second[..., 0] - sine * second[..., 1]
    y = first[..., 1] + sine * second[..., 0] + cosine * second[..., 1]
    return torch.stack((x, y, _wrap(first[..., 2] + second[..., 2])), dim=-1)


def _integrate_batch(relative: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    """World xy after each step. ``relative`` is ``(B, T, 3)``, ``origin`` is ``(B, 3)``."""
    phi = relative[..., 2]
    zeros = phi.new_zeros(phi.shape[0], 1)
    heading = origin[:, 2:3] + torch.cat((zeros, torch.cumsum(phi, dim=1)[:, :-1]), dim=1)
    cosine = torch.cos(heading)
    sine = torch.sin(heading)
    dx = cosine * relative[..., 0] - sine * relative[..., 1]
    dy = sine * relative[..., 0] + cosine * relative[..., 1]
    x = origin[:, 0:1] + torch.cumsum(dx, dim=1)
    y = origin[:, 1:2] + torch.cumsum(dy, dim=1)
    return torch.stack((x, y), dim=-1)


def _command(
    model: torch.nn.Module,
    counts: torch.Tensor,
    increments: torch.Tensor,
    commands: torch.Tensor | None = None,
) -> torch.Tensor:
    """Commanded (v, ω), raw ticks, or the wheel step, depending on the model."""
    if getattr(model, "reads_command", False):
        if commands is None:
            raise ValueError("this model reads commanded velocities, and none were passed")
        return commands
    if getattr(model, "reads_counts", False):
        return counts
    return increments


def _predicted_relative(
    model: torch.nn.Module,
    counts: torch.Tensor,
    increments: torch.Tensor,
    residual: bool,
    poses: torch.Tensor | None = None,
    commands: torch.Tensor | None = None,
    nu0: torch.Tensor | None = None,
    dt: float | None = None,
    hidden: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if getattr(model, "uses_velocity", False):
        if commands is None or nu0 is None or dt is None:
            raise ValueError("a velocity-state model needs the command, a seed twist, and dt")
        velocity = rollout_command_velocity(
            model,
            nu0,
            commands,
            dt,
            hidden=hidden,
            poses=poses if getattr(model, "uses_pose", False) else None,
        )
        twist = _body_twist(velocity, dt)
        return _se2_exp(twist), velocity
    features = _command(model, counts, increments, commands)
    if getattr(model, "uses_pose", False):
        if poses is None:
            raise ValueError("a pose-conditioned model needs the logged pose at each step")
        features = torch.cat((features, poses), dim=-1)
    if not residual:
        return _se2_exp(model(features)), None
    twist = torch.stack(
        (increments[..., 0], torch.zeros_like(increments[..., 0]), increments[..., 1]),
        dim=-1,
    )
    return _compose_relative(_se2_exp(twist), _se2_exp(model(features))), None


def _body_twist(velocity: torch.Tensor, dt: float) -> torch.Tensor:
    """Tangent step ``(v dt, v_y dt, ω dt)``. Two-wide ``ν`` has ``v_y = 0``."""
    if velocity.shape[-1] == 3:
        return torch.stack(
            (velocity[..., 0] * dt, velocity[..., 1] * dt, velocity[..., 2] * dt),
            dim=-1,
        )
    zeros = velocity.new_zeros(velocity.shape[:-1])
    return torch.stack((velocity[..., 0] * dt, zeros, velocity[..., 1] * dt), dim=-1)


def _measured_velocity(
    model: torch.nn.Module,
    increments: np.ndarray,
    vicon: np.ndarray,
    dt: float,
) -> np.ndarray:
    encoder = increments / dt
    if getattr(model, "has_sideslip", False):
        return np.stack((encoder[:, 0], vicon[:, 1], encoder[:, 1]), axis=1).astype(np.float32)
    if getattr(model, "seeds_vicon", False):
        return vicon[:, [0, 2]].astype(np.float32)
    return encoder.astype(np.float32)


def _half_width(dt: float, horizon_s: float) -> int:
    """Steps on each side of the center. The full window is ``2 * half + 1`` steps."""
    width = int(round(horizon_s / dt))
    if width % 2 == 0:
        width += 1
    if width < 3:
        raise ValueError(f"horizon {horizon_s} s is shorter than three steps at dt {dt}")
    return width // 2


def _centers(
    logs: list[Trajectory], horizon_s: float
) -> tuple[list[tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, int]], list[tuple[int, int]]]:
    """Per-log arrays and every ``(log, center)`` that has a full window."""
    prepared: list[tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, int]] = []
    pairs: list[tuple[int, int]] = []
    for index, log in enumerate(logs):
        n_steps = int(log.increments.shape[0])
        half = _half_width(log.dt, horizon_s)
        width = 2 * half + 1
        if n_steps < width:
            raise ValueError(
                f"{log.name} has {n_steps} steps, shorter than a {horizon_s:.0f} s window ({width} steps)"
            )
        counts = None if log.counts is None else np.asarray(log.counts, dtype=np.float32)
        poses = np.asarray(log.poses, dtype=np.float32)
        vicon = body_steps(poses)
        vicon_nu = (vicon / float(log.dt)).astype(np.float32)
        prepared.append(
            (
                counts,
                np.asarray(log.increments, dtype=np.float32),
                poses,
                kept_steps(log),
                half,
                np.asarray(log.commands, dtype=np.float32),
                float(log.dt),
                vicon_nu,
            )
        )
        for center in range(half, n_steps - half):
            pairs.append((index, center))
    return prepared, pairs


def _batch_mse(
    model: torch.nn.Module,
    prepared: list[tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, int]],
    pairs: list[tuple[int, int]],
    residual: bool,
    yaw_length: float | None = None,
) -> torch.Tensor:
    counts = []
    increments = []
    poses = []
    keep = []
    commands = []
    seeds = []
    measured_nu = []
    dts = []
    for index, center in pairs:
        count, inc, pose, kept, half, command, dt, vicon = prepared[index]
        if getattr(model, "reads_counts", False) and count is None:
            raise ValueError("this model reads raw encoder increments, and a log has none")
        if count is None:
            count = inc
        start = center - half
        counts.append(count[start : center + half + 1])
        increments.append(inc[start : center + half + 1])
        poses.append(pose[start : center + half + 2])
        keep.append(kept[start : center + half + 1])
        commands.append(command[start : center + half + 1])
        measured = _measured_velocity(model, inc, vicon, dt)
        seed = measured[start - 1] if start > 0 else measured[start]
        seeds.append(seed)
        measured_nu.append(measured[start : center + half + 1])
        dts.append(dt)
    device = _module_device(model)
    count = torch.as_tensor(np.stack(counts), device=device)
    step = torch.as_tensor(np.stack(increments), device=device)
    pose = torch.as_tensor(np.stack(poses), device=device)
    kept = torch.as_tensor(np.stack(keep), device=device)
    command = torch.as_tensor(np.stack(commands), device=device)
    nu0 = torch.as_tensor(np.stack(seeds), device=device)
    measured = torch.as_tensor(np.stack(measured_nu), device=device)
    dt = float(dts[0])
    hidden = None
    if getattr(model, "uses_hidden", False):
        warm = int(model.history_steps(dt))
        warms_nu = []
        warms_u = []
        for index, center in pairs:
            _count, inc, _pose, _kept, half, command_np, log_dt, vicon = prepared[index]
            start = center - half
            measured_np = _measured_velocity(model, inc, vicon, log_dt)
            nu_h, u_h = history_slice(measured_np, command_np, start, warm)
            warms_nu.append(nu_h)
            warms_u.append(u_h)
        hidden = model.warm_hidden(
            torch.as_tensor(np.stack(warms_nu), device=device),
            torch.as_tensor(np.stack(warms_u), device=device),
        )
    relative, velocity = _predicted_relative(
        model, count, step, residual, pose[:, :-1], command, nu0, dt, hidden
    )
    xy = _integrate_batch(relative, pose[:, 0])
    squared = (xy - pose[:, 1:, :2]) ** 2
    if yaw_length is not None:
        if yaw_length <= 0:
            raise ValueError(f"yaw_length must be positive, got {yaw_length}")
        predicted_yaw = pose[:, 0, 2:3] + torch.cumsum(relative[..., 2], dim=1)
        yaw_m = _wrap(predicted_yaw - pose[:, 1:, 2]) * float(yaw_length)
        squared = torch.cat((squared, yaw_m.unsqueeze(-1) ** 2), dim=-1)
    if not bool(kept.any()):
        raise ValueError("a 5 s batch has no steps inside the loss mask")
    loss = squared[kept].mean()
    if velocity is not None:
        speed = velocity[..., 0] - measured[..., 0]
        if velocity.shape[-1] == 3:
            slip = velocity[..., 1] - measured[..., 1]
            arc = (velocity[..., 2] - measured[..., 2]) * (0.5 * ROBOT_DIAMETER)
            extra = speed[kept] ** 2 + slip[kept] ** 2 + arc[kept] ** 2
        else:
            arc = (velocity[..., 1] - measured[..., 1]) * (0.5 * ROBOT_DIAMETER)
            extra = speed[kept] ** 2 + arc[kept] ** 2
        loss = loss + VELOCITY_WEIGHT * extra.mean()
    penalty = getattr(model, "asymmetry_penalty", None)
    if model.training and penalty is not None:
        loss = loss + penalty(count[kept])
    return loss


def _mean_mse(
    model: torch.nn.Module,
    prepared: list[tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, int]],
    pairs: list[tuple[int, int]],
    residual: bool,
    batch_size: int,
    yaw_length: float | None = None,
) -> torch.Tensor:
    total = torch.zeros(())
    seen = 0
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        total = total + _batch_mse(model, prepared, chunk, residual, yaw_length).detach() * len(chunk)
        seen += len(chunk)
    return total / seen


def train_open_loop(
    model: torch.nn.Module,
    train_logs: list[Trajectory],
    eval_logs: list[Trajectory],
    *,
    epochs: int = 40,
    lr: float = 1e-4,
    rollout_every: int = 5,
    grad_clip: float = 0.5,
    batch_size: int = 64,
    residual: bool = False,
    horizon_s: float = HORIZON_S,
    val_horizon_s: float = VAL_HORIZON_S,
    yaw_length: float | None = None,
    name: str = "unicycle",
    patience: int | None = None,
    min_delta: float = 0.0,
) -> TrainResult:
    """Fit ``model`` on open-loop windows and restore the best validation path.

    Each center step owns the window from ``horizon_s / 2`` before it to
    ``horizon_s / 2`` after it. The rollout starts at the true Vicon pose at
    the beginning of that window. ``residual=False`` integrates
    ``Exp(model(v, ω))``. ``residual=True`` integrates ``Z ∘ Exp(model(v, ω))``.

    ``patience`` stops the run after that many epochs in a row with no drop in
    the validation ``val_horizon_s`` position MSE. ``min_delta`` is the smallest
    drop that counts. That same 30 s validation MSE selects the checkpoint.
    ``eval_logs`` is validation, never the test set.
    """
    if epochs < 0:
        raise ValueError(f"epochs must be non-negative, got {epochs}")
    if horizon_s <= 0:
        raise ValueError(f"horizon_s must be positive, got {horizon_s}")
    if val_horizon_s <= 0:
        raise ValueError(f"val_horizon_s must be positive, got {val_horizon_s}")
    if rollout_every <= 0:
        raise ValueError(f"rollout_every must be positive, got {rollout_every}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if patience is not None and patience < 1:
        raise ValueError(f"patience must be positive, got {patience}")
    if min_delta < 0:
        raise ValueError(f"min_delta must be non-negative, got {min_delta}")

    train_data, train_pairs = _centers(train_logs, horizon_s)
    eval_data, eval_pairs = _centers(eval_logs, horizon_s)
    half = train_data[0][4]
    yaw_note = "" if yaw_length is None else f"  heading × {yaw_length:.3f} m/rad"
    val_stride = max(1, int(round(VAL_STRIDE_S / float(eval_logs[0].dt))))
    print(
        f"{name:8s}  {horizon_s:.0f} s train windows  {2 * half + 1} steps  "
        f"train {len(train_pairs)}  validation {len(eval_pairs)}  batch {batch_size}  "
        f"val {val_horizon_s:.0f} s mse  stride {val_stride}{yaw_note}",
        flush=True,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr) if trainable else None
    history: list[dict] = []
    best_error = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_objective = float("inf")
    best_val = float("inf")
    since_improvement = 0
    order = torch.arange(len(train_pairs))

    for epoch in range(epochs + 1):
        if epoch == 0:
            model.eval()
            with torch.no_grad():
                loss = _mean_mse(model, train_data, train_pairs, residual, batch_size, yaw_length)
        elif optimizer is None:
            model.eval()
            with torch.no_grad():
                loss = _mean_mse(model, train_data, train_pairs, residual, batch_size, yaw_length)
        else:
            model.train()
            order = order[torch.randperm(len(order))]
            total = torch.zeros(())
            seen = 0
            for start in range(0, len(order), batch_size):
                chosen = [train_pairs[int(i)] for i in order[start : start + batch_size]]
                batch_loss = _batch_mse(model, train_data, chosen, residual, yaw_length)
                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                total = total + batch_loss.detach() * len(chosen)
                seen += len(chosen)
            loss = total / seen

        with torch.no_grad():
            model.eval()
            val_loss = float(
                pooled_rollout_position_mse(
                    model,
                    eval_logs,
                    residual=residual,
                    horizon_s=val_horizon_s,
                    start_stride=val_stride,
                )
            )
        row = {
            "epoch": epoch,
            "train_loss": float(loss.detach()),
            "val_loss": val_loss,
        }
        selected = False
        if val_loss < best_objective:
            best_objective = val_loss
            best_error = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            selected = True
        stopping = False
        if patience is not None:
            if epoch == 0:
                best_val = val_loss
            elif val_loss < best_val - min_delta:
                best_val = val_loss
                since_improvement = 0
            else:
                since_improvement += 1
                stopping = since_improvement >= patience
        if epoch % rollout_every == 0 or epoch == epochs or stopping:
            error = float(held_out_rmse(model, eval_logs, residual=residual))
            row["held_out_position_rmse"] = error
            heading_note = ""
            if yaw_length is not None:
                heading = float(held_out_heading_rmse(model, eval_logs, residual=residual))
                row["held_out_heading_rmse"] = heading
                heading_note = f"  heading {np.degrees(heading):.2f} deg"
            mark = "  saved" if selected else ""
            rmse_note = f"  full val RMSE {error:.4f} m{heading_note}{mark}"
        else:
            row["held_out_position_rmse"] = float("nan")
            if yaw_length is not None:
                row["held_out_heading_rmse"] = float("nan")
            rmse_note = "  saved" if selected else ""
        print(
            f"{name:8s}  epoch {epoch:02d}  5s mse {row['train_loss']:.6e}  "
            f"val {val_horizon_s:.0f}s mse {val_loss:.6e}{rmse_note}",
            flush=True,
        )
        history.append(row)
        if stopping:
            print(
                f"{name:8s}  early stop at epoch {epoch:02d}  "
                f"validation {val_horizon_s:.0f} s mse flat for {patience} epochs",
                flush=True,
            )
            break
        if epoch == epochs:
            break

    model.load_state_dict(best_state)
    best_error = float(
        pooled_rollout_position_mse(
            model,
            eval_logs,
            residual=residual,
            horizon_s=val_horizon_s,
            start_stride=val_stride,
        )
    )
    print(
        f"{name:8s}  best epoch {best_epoch:02d}  "
        f"validation {val_horizon_s:.0f} s mse {best_error:.6e}  "
        f"selected by validation {val_horizon_s:.0f} s position MSE"
    )
    return TrainResult(history=history, best_epoch=best_epoch, best_error=best_error)


def epochs_for_same_budget(
    base_epochs: int,
    full_logs: list[Trajectory],
    reduced_logs: list[Trajectory],
    horizon_s: float = HORIZON_S,
) -> int:
    """Epochs that give ``reduced_logs`` about as many window updates as ``base_epochs`` on ``full_logs``.

    Shortening a log (or dropping collections) makes each epoch smaller.
    Scaling the epoch count by the ratio of 5 s training windows matches the
    original number of updates.
    """
    if base_epochs < 1:
        raise ValueError(f"base_epochs must be positive, got {base_epochs}")
    _, full_pairs = _centers(full_logs, horizon_s)
    _, reduced_pairs = _centers(reduced_logs, horizon_s)
    if not full_pairs or not reduced_pairs:
        raise ValueError("both training sets need at least one 5 s window")
    return max(1, int(round(base_epochs * len(full_pairs) / len(reduced_pairs))))


def save_checkpoint(
    name: str,
    model: torch.nn.Module,
    result: TrainResult,
    *,
    residual: bool,
    folder: Path = CHECKPOINT_DIR,
) -> None:
    """Overwrite ``name`` in ``runs/unicycle`` with the selected validation checkpoint."""
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / f"{name}.pt")
    payload = {
        "name": name,
        "residual": residual,
        "best_epoch": result.best_epoch,
        "best_error": result.best_error,
        "history": result.history,
    }
    penalty = getattr(model, "penalty", None)
    if penalty is not None:
        payload["penalty"] = float(penalty)
    tau = getattr(model, "tau", None)
    if tau is not None:
        payload["tau_v"] = float(tau[0])
        payload["tau_omega"] = float(tau[1])
    if getattr(model, "uses_hidden", False):
        payload["hidden"] = int(model.hidden_size)
        payload["history_s"] = float(model.history_s)
    scale = getattr(model, "delta_scale", None)
    if scale is None and name.endswith("_norm"):
        scale = getattr(model, "residual_limit", None)
    if scale is not None:
        payload["delta_v"] = float(scale[0])
        payload["delta_omega"] = float(scale[1])
    (folder / f"{name}.json").write_text(json.dumps(payload))
    print(f"{name:8s}  wrote {folder / name}.pt")


def load_checkpoint(name: str, *, folder: Path = CHECKPOINT_DIR) -> tuple[torch.nn.Module, TrainResult, bool]:
    """Load the saved weights and the loss history for ``name``."""
    from unicycle.models import (
        CanonicalModel,
        DynamicsFullState,
        DynamicsQuotient,
        DynamicsQuotientSymmetry,
        CommandCanonicalInstant,
        CommandCanonicalModel,
        CommandLagModel,
        CommandSymmetricFree,
        CommandSymmetricGRU,
        CommandSymmetricInstant,
        CommandSymmetricInstantFrontBack,
        CommandSymmetricInstantGRU,
        CommandSymmetricInstantNorm,
        CommandSymmetricInstantSlip,
        CommandSymmetricModel,
        FrontBackSymmetricModel,
        NaiveModel,
        EvenOddModel,
        ReversibleSymmetricModel,
        ResidualModel,
        SymmetricAsymModel,
        SymmetricModel,
    )

    classes = {
        "naive": NaiveModel,
        "canonical": CanonicalModel,
        "dynamics_fullstate": DynamicsFullState,
        "dynamics_quotient": DynamicsQuotient,
        "dynamics_quotient_mirror": DynamicsQuotient,
        "dynamics_quotient_symmetry": DynamicsQuotientSymmetry,
        "command_lag": CommandLagModel,
        "command_canonical": CommandCanonicalModel,
        "command_canonical_mirror": CommandCanonicalModel,
        "command_canonical_instant": CommandCanonicalInstant,
        "command_symmetric": CommandSymmetricModel,
        "command_symmetric_gru": CommandSymmetricGRU,
        "command_symmetric_instant": CommandSymmetricInstant,
        "command_symmetric_instant_gru": CommandSymmetricInstantGRU,
        "command_symmetric_instant_frontback": CommandSymmetricInstantFrontBack,
        "command_symmetric_instant_10s": CommandSymmetricInstant,
        "command_symmetric_instant_norm": CommandSymmetricInstantNorm,
        "command_symmetric_instant_slip": CommandSymmetricInstantSlip,
        "command_symmetric_free": CommandSymmetricFree,
        "symmetric": SymmetricModel,
        "reversible": ReversibleSymmetricModel,
        "front_back": FrontBackSymmetricModel,
        "even_odd": EvenOddModel,
        "symmetric_asym": SymmetricAsymModel,
        "residual": ResidualModel,
    }
    if name not in classes:
        raise KeyError(f"unknown checkpoint {name}")
    path = folder / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no saved {name} run in {folder}. Train that model first.")
    payload = json.loads(path.read_text())
    kwargs = {}
    if "penalty" in payload:
        kwargs["penalty"] = float(payload["penalty"])
    if "tau_v" in payload:
        kwargs["tau_v"] = float(payload["tau_v"])
        kwargs["tau_omega"] = float(payload["tau_omega"])
    if "delta_v" in payload:
        kwargs["delta_v"] = float(payload["delta_v"])
        kwargs["delta_omega"] = float(payload["delta_omega"])
    if "hidden" in payload:
        kwargs["hidden"] = int(payload["hidden"])
    if "history_s" in payload:
        kwargs["history_s"] = float(payload["history_s"])
    model = classes[name](**kwargs)
    model.load_state_dict(torch.load(folder / f"{name}.pt", weights_only=True))
    model.eval()
    result = TrainResult(
        history=payload["history"],
        best_epoch=int(payload["best_epoch"]),
        best_error=float(payload["best_error"]),
    )
    return model, result, bool(payload["residual"])
