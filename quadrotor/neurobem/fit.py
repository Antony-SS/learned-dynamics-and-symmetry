"""Load-or-train ParityMotorNet for NeuroBEM leftover experiments."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from quadrotor.canonical import PreparedFlight, canonical_dim, output_dim
from quadrotor.geometry import train_device
from quadrotor.neurobem.openloop import train_open_loop
from quadrotor.neurobem.sequence import ParityMotorNet

PARITY_WIDTH = 24
EPOCHS = 20


def n_params(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def unpack_state(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "state" in payload:
        return payload["state"]
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"unrecognized checkpoint type {type(payload)}")


class SymmetricModel(ParityMotorNet):
    """Official one-pass Z2×Z2 leftover. Same net on every dataset."""

    def __init__(self) -> None:
        super().__init__(
            canonical_dim(),
            output_dim(),
            hidden_per_type=PARITY_WIDTH,
        )


def load_or_train_symmetric(
    train_flights: list[PreparedFlight],
    val_flights: list[PreparedFlight],
    test_flights: list[PreparedFlight],
    folder: str | Path,
    *,
    seed: int = 0,
    device: str | torch.device = "cpu",
    epochs: int = EPOCHS,
    stride: int = 1,
    horizon: int = 50,
    warm_start: bool = False,
    warm_steps: int = 0,
    name: str = "symmetric",
) -> tuple[torch.nn.Module, list[dict]]:
    """Fit ParityMotorNet-24 if ``folder/name.pt`` is missing. Never overwrites."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.pt"
    history_path = folder / "results.json"
    eval_device = torch.device(device)
    model = SymmetricModel()
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(unpack_state(payload))
        history = payload.get("history", []) if isinstance(payload, dict) else []
        if not history and history_path.is_file():
            blob = json.loads(history_path.read_text())
            history = blob.get("models", {}).get(name, {}).get("history", [])
        model.to(eval_device)
        model.eval()
        print(f"loaded {name:13s}  params {n_params(model)}  from {path}")
        return model, history

    fit_device = train_device(eval_device)
    print(
        f"training {name:13s}  params {n_params(model)}  seed {seed}  "
        f"on {fit_device}  -> {path}",
        flush=True,
    )
    torch.manual_seed(seed)
    model.to(fit_device)
    result = train_open_loop(
        model,
        train_flights,
        val_flights,
        test_flights,
        epochs=epochs,
        lr=1e-3,
        rollout_every=5,
        batch_size=256,
        stride=stride,
        horizon=horizon,
        horizon_weight=0.0,
        warm_start=warm_start,
        warm_steps=warm_steps,
        name=name,
    )
    payload = {
        "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "history": result.history,
        "best_epoch": result.best_epoch,
        "best_error": result.best_error,
        "parameters": n_params(model),
        "seed": seed,
        "stride": stride,
        "horizon": horizon,
        "warm_start": warm_start,
        "warm_steps": warm_steps,
    }
    torch.save(payload, path)
    summary = {"models": {}}
    if history_path.is_file():
        summary = json.loads(history_path.read_text())
        summary.setdefault("models", {})
    summary["models"][name] = {
        "parameters": n_params(model),
        "best_epoch": result.best_epoch,
        "best_error": result.best_error,
        "seed": seed,
        "history": result.history,
    }
    history_path.write_text(json.dumps(summary, indent=2))
    model.to(eval_device)
    model.eval()
    return model, result.history
