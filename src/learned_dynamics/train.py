"""Full-batch supervised fit of a raw dynamics map.

The loss divides each label channel by its training-set standard deviation so a
small displacement is not drowned out by a velocity channel. Inputs are left in
physical units: standardizing coordinates would erase the geometry an
equivariant architecture is meant to use.

The checkpoint is the epoch with the lowest held-out open-loop position error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class TrainResult:
    history: list[dict]
    best_epoch: int
    best_error: float


def train_model(
    model: torch.nn.Module,
    features,
    targets,
    rollout_error: Callable[[torch.nn.Module], float],
    *,
    epochs: int = 25,
    lr: float = 1e-3,
    rollout_every: int = 5,
    name: str = "model",
    device: str = "cpu",
) -> TrainResult:
    """Fit ``model(features)`` to ``targets`` and restore the best rollout.

    ``rollout_error`` returns a position RMSE in meters. It is called on epoch
    0, every ``rollout_every`` epochs, and on the last epoch.
    """
    if epochs < 0:
        raise ValueError(f"epochs must be non-negative, got {epochs}")
    if rollout_every <= 0:
        raise ValueError(f"rollout_every must be positive, got {rollout_every}")

    model.to(device)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError(f"expected matching rank-2 batches, got {tuple(x.shape)} and {tuple(y.shape)}")
    scale = y.std(dim=0).clamp_min(1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict] = []
    best_error = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    for epoch in range(epochs + 1):
        model.train()
        prediction = model(x)
        loss = torch.mean(((prediction - y) / scale) ** 2)
        row = {"epoch": epoch, "train_loss": float(loss.detach())}
        if epoch % rollout_every == 0 or epoch == epochs:
            error = float(rollout_error(model))
            row["held_out_position_rmse"] = error
            mark = ""
            if error < best_error:
                best_error = error
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                mark = "  saved"
            print(
                f"{name:8s}  epoch {epoch:02d}  loss {row['train_loss']:.6e}  "
                f"held-out position RMSE {error:.4f} m{mark}",
                flush=True,
            )
        else:
            row["held_out_position_rmse"] = float("nan")
            print(f"{name:8s}  epoch {epoch:02d}  loss {row['train_loss']:.6e}", flush=True)
        history.append(row)
        if epoch == epochs:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.load_state_dict(best_state)
    print(f"{name:8s}  best epoch {best_epoch:02d}  held-out position RMSE {best_error:.4f} m")
    return TrainResult(history=history, best_epoch=best_epoch, best_error=best_error)
