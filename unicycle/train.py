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
from unicycle.data import MAX_LINEAR_VELOCITY, Trajectory, kept_steps
from unicycle.rollout import held_out_heading_rmse, held_out_rmse

# One folder, overwritten on each training run. Plot cells read these files.
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "runs" / "unicycle"

HORIZON_S = 5.0
# Wrapped heading enters the loss as meters of arc after one second at the
# linear speed cap. On the unicycle's 5 s windows that matches the RMS of one
# position component, so yaw does not drown xy and xy does not drown yaw.
HEADING_LENGTH = MAX_LINEAR_VELOCITY
_ANGLE_EPS = 1e-8


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


def _command(model: torch.nn.Module, counts: torch.Tensor, increments: torch.Tensor) -> torch.Tensor:
    """Raw tick increments for models that ask for them, otherwise the wheel step."""
    if getattr(model, "reads_counts", False):
        return counts
    return increments


def _predicted_relative(
    model: torch.nn.Module,
    counts: torch.Tensor,
    increments: torch.Tensor,
    residual: bool,
    poses: torch.Tensor | None = None,
) -> torch.Tensor:
    features = _command(model, counts, increments)
    if getattr(model, "uses_pose", False):
        if poses is None:
            raise ValueError("a pose-conditioned model needs the logged pose at each step")
        features = torch.cat((features, poses), dim=-1)
    if not residual:
        return _se2_exp(model(features))
    twist = torch.stack(
        (increments[..., 0], torch.zeros_like(increments[..., 0]), increments[..., 1]),
        dim=-1,
    )
    return _compose_relative(_se2_exp(twist), _se2_exp(model(features)))


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
        prepared.append(
            (
                counts,
                np.asarray(log.increments, dtype=np.float32),
                np.asarray(log.poses, dtype=np.float32),
                kept_steps(log),
                half,
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
    for index, center in pairs:
        count, inc, pose, kept, half = prepared[index]
        if getattr(model, "reads_counts", False) and count is None:
            raise ValueError("this model reads raw encoder increments, and a log has none")
        if count is None:
            count = inc
        counts.append(count[center - half : center + half + 1])
        increments.append(inc[center - half : center + half + 1])
        poses.append(pose[center - half : center + half + 2])
        keep.append(kept[center - half : center + half + 1])
    device = next(model.parameters()).device
    count = torch.as_tensor(np.stack(counts), device=device)
    step = torch.as_tensor(np.stack(increments), device=device)
    pose = torch.as_tensor(np.stack(poses), device=device)
    kept = torch.as_tensor(np.stack(keep), device=device)
    relative = _predicted_relative(model, count, step, residual, pose[:, :-1])
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
    the validation 5 s window loss. ``min_delta`` is the smallest drop that
    counts. With ``yaw_length``, that same yaw-inclusive validation loss also
    selects the checkpoint. Without it, full validation xy RMSE selects the
    checkpoint as before. ``eval_logs`` is validation, never the test set.
    """
    if epochs < 0:
        raise ValueError(f"epochs must be non-negative, got {epochs}")
    if horizon_s <= 0:
        raise ValueError(f"horizon_s must be positive, got {horizon_s}")
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
    print(
        f"{name:8s}  {horizon_s:.0f} s windows  {2 * half + 1} steps  "
        f"train {len(train_pairs)}  validation {len(eval_pairs)}  batch {batch_size}{yaw_note}",
        flush=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            val_loss = _mean_mse(model, eval_data, eval_pairs, residual, batch_size, yaw_length)
        row = {
            "epoch": epoch,
            "train_loss": float(loss.detach()),
            "val_loss": float(val_loss.detach()),
        }
        selected = False
        if yaw_length is not None and row["val_loss"] < best_objective:
            best_objective = row["val_loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            selected = True
        stopping = False
        if patience is not None:
            if epoch == 0:
                best_val = row["val_loss"]
            elif row["val_loss"] < best_val - min_delta:
                best_val = row["val_loss"]
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
            mark = ""
            if yaw_length is None and error < best_error:
                best_error = error
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                selected = True
            if selected:
                mark = "  saved"
            rmse_note = f"  validation position RMSE {error:.4f} m{heading_note}{mark}"
        else:
            row["held_out_position_rmse"] = float("nan")
            if yaw_length is not None:
                row["held_out_heading_rmse"] = float("nan")
            rmse_note = ""
        print(
            f"{name:8s}  epoch {epoch:02d}  5s mse {row['train_loss']:.6e}  "
            f"val {row['val_loss']:.6e}{rmse_note}",
            flush=True,
        )
        history.append(row)
        if stopping:
            print(
                f"{name:8s}  early stop at epoch {epoch:02d}  "
                f"validation 5 s mse flat for {patience} epochs",
                flush=True,
            )
            break
        if epoch == epochs:
            break

    model.load_state_dict(best_state)
    best_error = float(held_out_rmse(model, eval_logs, residual=residual))
    criterion = "yaw-inclusive 5 s validation MSE" if yaw_length is not None else "validation position RMSE"
    print(
        f"{name:8s}  best epoch {best_epoch:02d}  "
        f"validation position RMSE {best_error:.4f} m  selected by {criterion}"
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
    (folder / f"{name}.json").write_text(json.dumps(payload))
    print(f"{name:8s}  wrote {folder / name}.pt")


def load_checkpoint(name: str, *, folder: Path = CHECKPOINT_DIR) -> tuple[torch.nn.Module, TrainResult, bool]:
    """Load the saved weights and the loss history for ``name``."""
    from unicycle.models import (
        CanonicalModel,
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
    model = classes[name](**kwargs)
    model.load_state_dict(torch.load(folder / f"{name}.pt", weights_only=True))
    model.eval()
    result = TrainResult(
        history=payload["history"],
        best_epoch=int(payload["best_epoch"]),
        best_error=float(payload["best_error"]),
    )
    return model, result, bool(payload["residual"])
