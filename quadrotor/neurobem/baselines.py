"""NeuroBEM published TCN: one-step force/torque, then integrate 0.5 s."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from quadrotor.canonical import PreparedFlight
from quadrotor.neurobem.data import HORIZON_STEPS
from quadrotor.evaluate import benchmark_mae
from quadrotor.geometry import quat_to_matrix
from quadrotor.neurobem.data import DEFAULT_ROOT, DOWNSAMPLE, _flight_id, _segment_id
from quadrotor.neurobem.tcn import NeuroBEMTCN
from quadrotor.train import _apply, _so3_log, module_device

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "runs" / "neurobem_tcn_100hz"
MASS = 0.772
GRAVITY = 9.80665
INERTIA = (0.0025, 0.0021, 0.0043)
HISTORY = NeuroBEMTCN.history_steps


def _load_tcn_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, PreparedFlight]:
    frame = pd.read_csv(path).iloc[::DOWNSAMPLE].reset_index(drop=True)
    quaternion = frame[["quat x", "quat y", "quat z", "quat w"]].to_numpy(np.float32)
    rotation = quat_to_matrix(quaternion).astype(np.float32)
    velocity_body = frame[["vel x", "vel y", "vel z"]].to_numpy(np.float32)
    velocity_world = np.einsum("nij,nj->ni", rotation, velocity_body).astype(np.float32)
    motors = frame[["mot 2", "mot 1", "mot 3", "mot 4"]].to_numpy(np.float32)
    omega = frame[["ang vel x", "ang vel y", "ang vel z"]].to_numpy(np.float32)
    acc = frame[["acc x", "acc y", "acc z"]].to_numpy(np.float32)
    alpha = frame[["ang acc x", "ang acc y", "ang acc z"]].to_numpy(np.float32)
    time = frame["t"].to_numpy(np.float64)
    dt = np.diff(time).astype(np.float32)
    force = MASS * acc
    torque = alpha * np.asarray(INERTIA, dtype=np.float32) + np.cross(
        omega, omega * np.asarray(INERTIA, dtype=np.float32)
    )
    labels = np.concatenate((force, torque), axis=1).astype(np.float32)
    features = np.concatenate((omega, velocity_body, motors), axis=1).astype(np.float32)
    flight = PreparedFlight(
        name=_segment_id(path),
        position=frame[["pos x", "pos y", "pos z"]].to_numpy(np.float32),
        rotation=rotation,
        velocity_world=velocity_world,
        omega_body=omega,
        pwm=motors[:-1],
        voltage=frame["vbat"].to_numpy(np.float32)[:-1],
        dt=dt,
        attitude="Vicon",
    )
    return features, labels, flight


def load_tcn_split(
    root: str | Path | None = None,
    *,
    seed: int = 0,
) -> tuple[list[tuple[np.ndarray, np.ndarray, PreparedFlight]], ...]:
    root = DEFAULT_ROOT if root is None else Path(root)
    paths = sorted((root / "processed_data").glob("merged_*.csv"))
    test_ids = {
        line.strip()
        for line in (root / "testset.txt").read_text().splitlines()
        if line.strip()
    }
    test_paths = [path for path in paths if _segment_id(path) in test_ids]
    remaining = [path for path in paths if _segment_id(path) not in test_ids]
    by_flight: dict[str, list[Path]] = {}
    for path in remaining:
        by_flight.setdefault(_flight_id(path), []).append(path)
    flight_ids = sorted(by_flight)
    shuffled = [
        flight_ids[index]
        for index in np.random.default_rng(seed).permutation(len(flight_ids))
    ]
    n_validation = max(1, int(round(0.20 * len(shuffled))))
    validation_ids = set(shuffled[:n_validation])
    validation_paths = [
        path for flight in validation_ids for path in sorted(by_flight[flight])
    ]
    train_paths = [
        path
        for flight in shuffled[n_validation:]
        for path in sorted(by_flight[flight])
    ]
    return (
        [_load_tcn_arrays(path) for path in train_paths],
        [_load_tcn_arrays(path) for path in validation_paths],
        [_load_tcn_arrays(path) for path in test_paths],
    )


class _ForceWindows(Dataset):
    def __init__(
        self,
        split: list[tuple[np.ndarray, np.ndarray, PreparedFlight]],
        stride: int,
    ) -> None:
        self.split = split
        self.stride = stride
        self.counts = [
            max(0, (len(features) - HISTORY) // stride + 1) for features, _labels, _flight in split
        ]
        self.ends = np.cumsum(self.counts).tolist()

    def __len__(self) -> int:
        return int(self.ends[-1]) if self.ends else 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        flight_index = int(np.searchsorted(self.ends, index, side="right"))
        previous = 0 if flight_index == 0 else self.ends[flight_index - 1]
        target = HISTORY - 1 + (index - previous) * self.stride
        features, labels, _flight = self.split[flight_index]
        return (
            torch.as_tensor(features[target - HISTORY + 1 : target + 1]),
            torch.as_tensor(labels[target]),
        )


class TCNDynamics(torch.nn.Module):
    """Paper TCN-medium: predict body force/torque, then the same SE(3) integrator."""

    reads_pose = False

    def __init__(self, tcn: NeuroBEMTCN | None = None) -> None:
        super().__init__()
        self.tcn = NeuroBEMTCN() if tcn is None else tcn
        self.register_buffer("inertia", torch.tensor(INERTIA, dtype=torch.float32))
        self.register_buffer("up", torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32))

    def initial_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(batch_size, HISTORY, NeuroBEMTCN.input_dim)

    def _features(
        self,
        rotation: torch.Tensor,
        velocity_world: torch.Tensor,
        omega: torch.Tensor,
        pwm: torch.Tensor,
    ) -> torch.Tensor:
        velocity_body = torch.bmm(
            rotation.transpose(1, 2), velocity_world.unsqueeze(-1)
        ).squeeze(-1)
        return torch.cat((omega, velocity_body, pwm), dim=-1)

    def prime_hidden(self, flight: PreparedFlight, start: int, device: torch.device) -> torch.Tensor:
        velocity_body = np.einsum(
            "nji,nj->ni",
            flight.rotation[start - HISTORY + 1 : start + 1],
            flight.velocity_world[start - HISTORY + 1 : start + 1],
        )
        features = np.concatenate(
            (
                flight.omega_body[start - HISTORY + 1 : start + 1],
                velocity_body,
                flight.pwm[start - HISTORY + 1 : start + 1],
            ),
            axis=1,
        )
        return torch.as_tensor(features[None], device=device)

    def advance(
        self,
        position: torch.Tensor,
        rotation: torch.Tensor,
        velocity_world: torch.Tensor,
        omega_body: torch.Tensor,
        pwm: torch.Tensor,
        dt: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._features(rotation, velocity_world, omega_body, pwm)
        hidden = torch.cat((hidden[:, 1:], features.unsqueeze(1)), dim=1)
        force_torque = self.tcn(hidden)
        force, torque = force_torque[:, :3], force_torque[:, 3:]
        gravity_body = torch.bmm(
            rotation.transpose(1, 2),
            self.up.expand(rotation.shape[0], 3).unsqueeze(-1),
        ).squeeze(-1)
        linear = force / MASS - GRAVITY * gravity_body
        angular = (torque - torch.linalg.cross(omega_body, omega_body * self.inertia)) / self.inertia
        position, rotation, velocity_world, omega_body = _apply(
            position,
            rotation,
            velocity_world,
            omega_body,
            torch.cat((linear, angular), dim=-1),
            dt,
        )
        return position, rotation, velocity_world, omega_body, hidden


def _tcn_loss(model: NeuroBEMTCN, history: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target = (labels - model.output_center) / model.output_scale
    error = model.normalized_prediction(history) - target
    weights = error.new_tensor((1.0, 1.0, 3.0, 1.0, 1.0, 1.0))
    return (error.square() * weights).sum(dim=-1).mean()


def _mean_loss(model: NeuroBEMTCN, loader: DataLoader, device: torch.device) -> float:
    total = 0.0
    seen = 0
    model.eval()
    with torch.no_grad():
        for history, labels in loader:
            history, labels = history.to(device), labels.to(device)
            total += float(_tcn_loss(model, history, labels)) * len(history)
            seen += len(history)
    return total / max(seen, 1)


def train_tcn(
    *,
    folder: str | Path | None = None,
    seed: int = 0,
    epochs: int = 20,
    device: str | torch.device = "cpu",
    stride: int = 4,
) -> tuple[TCNDynamics, list[dict]]:
    folder = CHECKPOINT_DIR if folder is None else Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "tcn.pt"
    train, validation, _test = load_tcn_split(seed=seed)
    requested = torch.device(device)
    if requested.type == "cpu" and torch.backends.mps.is_available():
        requested = torch.device("mps")
    torch.manual_seed(seed)
    tcn = NeuroBEMTCN().to(requested)
    inputs = torch.as_tensor(np.concatenate([features for features, _labels, _flight in train]))
    outputs = torch.as_tensor(np.concatenate([labels for _features, labels, _flight in train]))
    tcn.set_normalization(
        inputs.mean(0).to(requested),
        inputs.std(0).to(requested),
        outputs.mean(0).to(requested),
        outputs.std(0).to(requested),
    )
    train_loader = DataLoader(_ForceWindows(train, stride), batch_size=1024, shuffle=True)
    val_loader = DataLoader(_ForceWindows(validation, stride), batch_size=1024)
    optimizer = torch.optim.Adam(tcn.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history: list[dict] = []
    for epoch in range(epochs + 1):
        if epoch:
            tcn.train()
            total = 0.0
            seen = 0
            for features, labels in train_loader:
                features, labels = features.to(requested), labels.to(requested)
                loss = _tcn_loss(tcn, features, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(features)
                seen += len(features)
            train_loss = total / max(seen, 1)
            scheduler.step()
        else:
            train_loss = _mean_loss(tcn, train_loader, requested)
        val_loss = _mean_loss(tcn, val_loader, requested)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in tcn.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        print(
            f"tcn        epoch {epoch:03d}  train {train_loss:.6f}  val {val_loss:.6f}"
            f"{' saved' if epoch == best_epoch else ''}",
            flush=True,
        )
    assert best_state is not None
    tcn.load_state_dict(best_state)
    payload = {
        "state": best_state,
        "history": history,
        "best_epoch": best_epoch,
        "best_error": best_loss,
        "parameters": sum(parameter.numel() for parameter in tcn.parameters()),
        "seed": seed,
    }
    torch.save(payload, path)
    (folder / "results.json").write_text(json.dumps({"models": {"tcn": payload | {"state": None}}}, indent=2))
    model = TCNDynamics(tcn)
    model.to(requested)
    model.eval()
    return model, history


def load_or_train_tcn(
    *,
    folder: str | Path | None = None,
    seed: int = 0,
    epochs: int = 20,
    device: str | torch.device = "cpu",
) -> tuple[TCNDynamics, list[dict]]:
    folder = CHECKPOINT_DIR if folder is None else Path(folder)
    path = folder / "tcn.pt"
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tcn = NeuroBEMTCN()
        state = payload["state"] if isinstance(payload, dict) and "state" in payload else payload
        tcn.load_state_dict(state)
        history = payload.get("history", []) if isinstance(payload, dict) else []
        model = TCNDynamics(tcn)
        model.to(device)
        model.eval()
        print(f"loaded tcn            params {sum(p.numel() for p in tcn.parameters())}  from {path}")
        return model, history
    return train_tcn(folder=folder, seed=seed, epochs=epochs, device=device)


def score_tcn(
    model: TCNDynamics,
    flights: list[PreparedFlight],
    *,
    start_stride: int = 5,
    start_offset: int = HISTORY - 1,
    horizon: int = HORIZON_STEPS,
    batch_size: int = 128,
) -> dict[str, dict[str, float]]:
    """Open-loop MAE after priming the TCN with 20 steps of Vicon history."""
    starts = [
        (flight_index, start)
        for flight_index, flight in enumerate(flights)
        for start in range(start_offset, flight.n_steps - horizon + 1, start_stride)
    ]
    if not starts:
        raise ValueError("no TCN window is long enough")
    device = module_device(model)
    totals = {
        name: torch.zeros(horizon, dtype=torch.float64)
        for name in ("position", "velocity", "rotation", "omega")
    }
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch = starts[offset : offset + batch_size]
            hidden = torch.cat(
                [model.prime_hidden(flights[i], start, device) for i, start in batch],
                dim=0,
            )
            pwm = torch.stack(
                [torch.as_tensor(flights[i].pwm[start : start + horizon]) for i, start in batch]
            ).to(device)
            dt = torch.stack(
                [torch.as_tensor(flights[i].dt[start : start + horizon]) for i, start in batch]
            ).to(device)
            position = torch.stack(
                [torch.as_tensor(flights[i].position[start]) for i, start in batch]
            ).to(device)
            rotation = torch.stack(
                [torch.as_tensor(flights[i].rotation[start]) for i, start in batch]
            ).to(device)
            velocity = torch.stack(
                [torch.as_tensor(flights[i].velocity_world[start]) for i, start in batch]
            ).to(device)
            omega = torch.stack(
                [torch.as_tensor(flights[i].omega_body[start]) for i, start in batch]
            ).to(device)
            hats = {name: [] for name in totals}
            for step in range(horizon):
                position, rotation, velocity, omega, hidden = model.advance(
                    position, rotation, velocity, omega, pwm[:, step], dt[:, step], hidden
                )
                true_p = torch.stack(
                    [torch.as_tensor(flights[i].position[start + step + 1]) for i, start in batch]
                ).to(device)
                true_r = torch.stack(
                    [torch.as_tensor(flights[i].rotation[start + step + 1]) for i, start in batch]
                ).to(device)
                true_v = torch.stack(
                    [torch.as_tensor(flights[i].velocity_world[start + step + 1]) for i, start in batch]
                ).to(device)
                true_w = torch.stack(
                    [torch.as_tensor(flights[i].omega_body[start + step + 1]) for i, start in batch]
                ).to(device)
                relative = torch.bmm(true_r.transpose(1, 2), rotation)
                hats["position"].append(torch.linalg.vector_norm(position - true_p, dim=-1))
                hats["velocity"].append(torch.linalg.vector_norm(velocity - true_v, dim=-1))
                hats["rotation"].append(torch.linalg.vector_norm(_so3_log(relative), dim=-1))
                hats["omega"].append(torch.linalg.vector_norm(omega - true_w, dim=-1))
            for name, series in hats.items():
                totals[name] += torch.stack(series, dim=1).sum(dim=0).cpu().double()
    curves = {name: values / len(starts) for name, values in totals.items()}
    return {
        name: {
            "h=1": float(curve[0]),
            "h=10": float(curve[9]),
            "h=50": float(curve[49]),
            "h=100": float(curve[99]),
            "h=200": float(curve[199]),
        }
        for name, curve in curves.items()
    }


def score_ours_aligned(
    model: torch.nn.Module,
    flights: list[PreparedFlight],
    *,
    start_stride: int = 5,
) -> dict[str, dict[str, float]]:
    metrics = benchmark_mae(
        model,
        flights,
        horizon=HORIZON_STEPS,
        start_stride=start_stride,
        start_offset=HISTORY - 1,
        warm_start=True,
        warm_steps=HISTORY,
    )
    return {
        name: {
            "h=1": float(curve[0]),
            "h=10": float(curve[9]),
            "h=50": float(curve[49]),
            "h=100": float(curve[99]),
            "h=200": float(curve[199]),
        }
        for name, curve in metrics.items()
    }
