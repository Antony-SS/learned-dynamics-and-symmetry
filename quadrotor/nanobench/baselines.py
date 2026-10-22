"""Portable NanoBench baselines. Mixer physics needs rad/s; these logs are PWM."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from quadrotor.canonical import HORIZON_STEPS, PreparedFlight
from quadrotor.fit import n_params, train_device, unpack_state
from quadrotor.idsia.baselines import ResidualQuadModel, _windows, fit_scalers

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "runs" / "nanobench_residual"


def train_residual(
    train_flights: list[PreparedFlight],
    val_flights: list[PreparedFlight],
    *,
    folder: str | Path | None = None,
    seed: int = 0,
    epochs: int = 20,
    horizon: int = HORIZON_STEPS,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> tuple[ResidualQuadModel, list[dict]]:
    folder = CHECKPOINT_DIR if folder is None else Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    device = train_device(device)
    torch.manual_seed(seed)
    x_mean, x_scale, u_mean, u_scale = fit_scalers(train_flights)
    model = ResidualQuadModel(hidden_dim=64, num_layers=5)
    model.set_scalers(x_mean, x_scale, u_mean, u_scale)
    model.to(device)
    train_x, train_u, train_y = _windows(train_flights, horizon, stride=1)
    val_x, val_u, val_y = _windows(val_flights, horizon, stride=1)

    def _norm(x, u, y):
        return (x - x_mean) / x_scale, (u - u_mean) / u_scale, (y - x_mean) / x_scale

    train_x, train_u, train_y = _norm(train_x, train_u, train_y)
    val_x, val_u, val_y = _norm(val_x, val_u, val_y)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best = None
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict] = []
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            order = torch.randperm(len(train_x))
            train_loss = 0.0
            seen = 0
            for start in range(0, len(order), batch_size):
                index = order[start : start + batch_size]
                pred = model(train_x[index].to(device), train_u[index].to(device))
                loss = (pred - train_y[index].to(device)).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += float(loss) * len(index)
                seen += len(index)
            train_loss = train_loss / max(seen, 1)
        else:
            model.eval()
            with torch.no_grad():
                pred = model(train_x[:batch_size].to(device), train_u[:batch_size].to(device))
                train_loss = float((pred - train_y[:batch_size].to(device)).pow(2).mean())
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            counted = 0
            for start in range(0, len(val_x), batch_size):
                stop = start + batch_size
                pred = model(val_x[start:stop].to(device), val_u[start:stop].to(device))
                val_loss += float((pred - val_y[start:stop].to(device)).pow(2).mean()) * (stop - start)
                counted += stop - start
        val_loss /= max(counted, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        mark = ""
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            mark = "  saved"
        print(
            f"residual   epoch {epoch:03d}  train {train_loss:.6f}  val {val_loss:.6f}{mark}",
            flush=True,
        )
    assert best is not None
    model.load_state_dict(best)
    payload = {
        "state": best,
        "history": history,
        "best_epoch": best_epoch,
        "best_error": best_loss,
        "parameters": n_params(model),
        "seed": seed,
    }
    torch.save(payload, folder / "residual.pt")
    (folder / "results.json").write_text(json.dumps({"models": {"residual": {k: v for k, v in payload.items() if k != "state"}}}, indent=2))
    model.to(device)
    model.eval()
    return model, history


def load_or_train_residual(
    train_flights: list[PreparedFlight],
    val_flights: list[PreparedFlight],
    *,
    folder: str | Path | None = None,
    seed: int = 0,
    epochs: int = 20,
    device: str | torch.device = "cpu",
) -> tuple[ResidualQuadModel, list[dict]]:
    folder = CHECKPOINT_DIR if folder is None else Path(folder)
    path = folder / "residual.pt"
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = ResidualQuadModel(hidden_dim=64, num_layers=5)
        x_mean, x_scale, u_mean, u_scale = fit_scalers(train_flights)
        model.set_scalers(x_mean, x_scale, u_mean, u_scale)
        model.load_state_dict(unpack_state(payload), strict=False)
        model.to(device)
        model.eval()
        history = payload.get("history", []) if isinstance(payload, dict) else []
        print(f"loaded residual      params {n_params(model)}  from {path}")
        return model, history
    return train_residual(
        train_flights,
        val_flights,
        folder=folder,
        seed=seed,
        epochs=epochs,
        device=device,
    )
