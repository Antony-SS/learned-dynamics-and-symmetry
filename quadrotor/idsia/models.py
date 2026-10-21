"""IDSIA models we train: naive, canonical, symmetric, physics + symmetric GRU.

Naive and canonical are an unconstrained GRU sized to the parity net (~18k).
Symmetric and the residual on frozen IDSIA physics are ``ParityMotorNet``
with ``hidden_per_type=24`` (17,977 weights): one pass, both airframe axes.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from quadrotor.canonical import PreparedFlight, absolute_dim, canonical_dim, output_dim
from quadrotor.idsia.baselines import PhysQuadModel
from quadrotor.sequence import ParityMotorNet
from quadrotor.train import _features, _so3_exp, _so3_log, train_open_loop

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "runs" / "idsia_parity18k"
INCLUDE_VOLTAGE = False
PARITY_WIDTH = 24
PLAIN_HIDDEN = 60
EPOCHS = 20
OURS = ("naive", "canonical", "symmetric", "phys_sym_gru")


class PlainGRU(torch.nn.Module):
    """Unconstrained GRU + MLP head, ~18k weights. No airframe symmetry."""

    reads_pose = False
    mirror = False

    def __init__(self, in_dim: int, hidden_dim: int = PLAIN_HIDDEN) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gru = torch.nn.GRUCell(in_dim, self.hidden_dim)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(self.hidden_dim, output_dim())
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.register_buffer("input_center", torch.zeros(in_dim, dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones(in_dim, dtype=torch.float32))
        self.register_buffer(
            "acceleration_limit",
            torch.tensor((40.0, 40.0, 40.0, 400.0, 400.0, 400.0), dtype=torch.float32),
        )

    def set_input_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        if center.shape != self.input_center.shape or scale.shape != self.input_scale.shape:
            raise ValueError(
                f"normalization must have shape {tuple(self.input_center.shape)}"
            )
        with torch.no_grad():
            self.input_center.copy_(center.to(self.input_center))
            self.input_scale.copy_(scale.to(self.input_scale).clamp_min(1e-6))

    def initial_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(batch_size, self.hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.initial_hidden(features.shape[0], features)
        acceleration, _ = self.step(features, hidden)
        return acceleration

    def _plain_step(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features - self.input_center) / self.input_scale
        hidden = self.gru(normalized, hidden)
        acceleration = torch.tanh(self.head(self.trunk(hidden)))
        return acceleration * self.acceleration_limit, hidden

    def step(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._plain_step(features, hidden)


class NaiveModel(PlainGRU):
    reads_pose = True

    def __init__(self) -> None:
        super().__init__(absolute_dim(INCLUDE_VOLTAGE))


class CanonicalModel(PlainGRU):
    def __init__(self) -> None:
        super().__init__(canonical_dim(INCLUDE_VOLTAGE))


class SymmetricModel(ParityMotorNet):
    def __init__(self) -> None:
        super().__init__(
            canonical_dim(INCLUDE_VOLTAGE),
            output_dim(),
            hidden_per_type=PARITY_WIDTH,
        )


class LongitudinalModel(PlainGRU):
    """Same ~18k GRU, orbit-averaged over the longitudinal (left-right) mirror.

    Front-back is not imposed. Hidden state is two copies: identity and LR.
    """

    longitudinal_mirror = True

    def __init__(self) -> None:
        super().__init__(canonical_dim(INCLUDE_VOLTAGE))

    def initial_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(batch_size, 2, self.hidden_dim)

    def step(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from quadrotor.sequence import _mirror_acceleration, _mirror_features

        first, hidden_first = self._plain_step(features, hidden[:, 0])
        mirrored, hidden_mirror = self._plain_step(
            _mirror_features(features, "lr"), hidden[:, 1]
        )
        mirrored = _mirror_acceleration(mirrored, "lr")
        return 0.5 * (first + mirrored), torch.stack((hidden_first, hidden_mirror), dim=1)


class SymmetricHoldModel(SymmetricModel):
    """Previous twist plus a symmetric residual, then SE(3) from the new twist.

    ``ν_{k+1} = ν_k + f_θ(ν_k, u_k)``. Pose is integrated with ``ν_{k+1}``.
    There is no velocity command to anchor to, so the previous Vicon/predicted
    twist plays that role. Zero ``f_θ`` coasts.
    """

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
        features = _features(self, position, rotation, velocity_world, omega_body, pwm)
        residual, hidden = self.step(features, hidden)
        step = dt.reshape(-1, 1)
        delta_v = torch.bmm(
            rotation, (residual[:, 0:3] * step).unsqueeze(-1)
        ).squeeze(-1)
        velocity_world = velocity_world + delta_v
        omega_body = omega_body + residual[:, 3:6] * step
        position = position + velocity_world * step
        rotation = torch.bmm(rotation, _so3_exp(omega_body * step))
        return position, rotation, velocity_world, omega_body, hidden


class SymmetricZModel(ParityMotorNet):
    """Same one-pass airframe symmetry, plus world z for ground effect."""

    reads_height = True

    def __init__(self) -> None:
        super().__init__(
            canonical_dim(INCLUDE_VOLTAGE) + 1,
            output_dim(),
            hidden_per_type=PARITY_WIDTH,
        )


class PhysSymGRU(torch.nn.Module):
    """Frozen IDSIA physics plus a one-pass symmetric GRU residual."""

    reads_pose = False
    mirror = True

    def __init__(self) -> None:
        super().__init__()
        self.phys = PhysQuadModel(zero_omega_dot=True)
        for parameter in self.phys.parameters():
            parameter.requires_grad = False
        self.neural = SymmetricModel()
        in_dim = canonical_dim(INCLUDE_VOLTAGE)
        self.register_buffer("input_center", torch.zeros(in_dim, dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones(in_dim, dtype=torch.float32))

    def set_input_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        self.neural.set_input_normalization(center, scale)
        with torch.no_grad():
            self.input_center.copy_(center.to(self.input_center))
            self.input_scale.copy_(scale.to(self.input_scale).clamp_min(1e-6))

    def initial_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return self.neural.initial_hidden(batch_size, reference)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.neural(features)

    def step(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.neural.step(features, hidden)

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
        state = torch.cat((position, velocity_world, _so3_log(rotation), omega_body), dim=-1)
        with torch.no_grad():
            next_state = self.phys.one_step(state, pwm)
        pos_p, vel_p, so3_p, omega_p = next_state.split((3, 3, 3, 3), dim=-1)
        rotation_p = _so3_exp(so3_p)
        up = rotation.new_tensor((0.0, 0.0, 1.0)).expand(rotation.shape[0], 3)
        rotation_t = rotation.transpose(1, 2)
        features = torch.cat(
            (
                torch.bmm(rotation_t, up.unsqueeze(-1)).squeeze(-1),
                torch.bmm(rotation_t, velocity_world.unsqueeze(-1)).squeeze(-1),
                omega_body,
                pwm,
            ),
            dim=-1,
        )
        acceleration, hidden = self.neural.step(features, hidden)
        step = dt.reshape(-1, 1)
        delta_pos = torch.bmm(
            rotation, (0.5 * acceleration[:, 0:3] * step.square()).unsqueeze(-1)
        ).squeeze(-1)
        delta_vel = torch.bmm(
            rotation, (acceleration[:, 0:3] * step).unsqueeze(-1)
        ).squeeze(-1)
        delta_omega = acceleration[:, 3:6] * step
        delta_rot = acceleration[:, 3:6] * step.square() * 0.5
        return (
            pos_p + delta_pos,
            torch.bmm(rotation_p, _so3_exp(delta_rot)),
            vel_p + delta_vel,
            omega_p + delta_omega,
            hidden,
        )


def _empty(name: str) -> torch.nn.Module:
    if name == "naive":
        return NaiveModel()
    if name == "canonical":
        return CanonicalModel()
    if name == "symmetric":
        return SymmetricModel()
    if name == "longitudinal":
        return LongitudinalModel()
    if name == "phys_sym_gru":
        return PhysSymGRU()
    raise KeyError(name)


def n_params(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _train_device(requested: str | torch.device) -> torch.device:
    requested = torch.device(requested)
    if requested.type == "cpu" and torch.backends.mps.is_available():
        return torch.device("mps")
    return requested


def _checkpoint_path(name: str, folder: Path) -> Path:
    return folder / f"{name}.pt"


def _history_path(folder: Path) -> Path:
    return folder / "results.json"


def load_ours(
    folder: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.nn.Module], dict[str, list[dict]]]:
    folder = CHECKPOINT_DIR if folder is None else Path(folder)
    histories: dict[str, list[dict]] = {}
    summary = _history_path(folder)
    if summary.is_file():
        blob = json.loads(summary.read_text())
        for name, record in blob.get("models", {}).items():
            histories[name] = record.get("history", [])
    loaded: dict[str, torch.nn.Module] = {}
    for name in OURS:
        path = _checkpoint_path(name, folder)
        if not path.is_file():
            raise FileNotFoundError(path)
        model = _empty(name)
        payload = torch.load(path, map_location=device, weights_only=False)
        state = payload["state"] if isinstance(payload, dict) and "state" in payload else payload
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        loaded[name] = model
        if name not in histories and isinstance(payload, dict):
            histories[name] = payload.get("history", [])
    return loaded, histories


def load_or_train_ours(
    train_flights: list[PreparedFlight],
    val_flights: list[PreparedFlight],
    test_flights: list[PreparedFlight],
    *,
    device: str | torch.device = "cpu",
    epochs: int = EPOCHS,
    folder: str | Path | None = None,
) -> tuple[dict[str, torch.nn.Module], dict[str, list[dict]]]:
    """Load the four IDSIA nets we train, or fit any that are missing."""
    folder = CHECKPOINT_DIR if folder is None else Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    train_device = _train_device(device)
    models: dict[str, torch.nn.Module] = {}
    histories: dict[str, list[dict]] = {}
    summary: dict = {"models": {}}
    if _history_path(folder).is_file():
        summary = json.loads(_history_path(folder).read_text())

    for name in OURS:
        path = _checkpoint_path(name, folder)
        model = _empty(name)
        if path.is_file():
            payload = torch.load(path, map_location=device, weights_only=False)
            state = payload["state"] if isinstance(payload, dict) and "state" in payload else payload
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            history = []
            if isinstance(payload, dict):
                history = payload.get("history", [])
            if not history:
                history = summary.get("models", {}).get(name, {}).get("history", [])
            print(f"loaded {name:13s}  params {n_params(model)}  from {path}")
        else:
            print(
                f"training {name:13s}  params {n_params(model)}  "
                f"on {train_device}  -> {path}",
                flush=True,
            )
            torch.manual_seed(0)
            model.to(train_device)
            result = train_open_loop(
                model,
                train_flights,
                val_flights,
                test_flights,
                epochs=epochs,
                lr=1e-3,
                rollout_every=5,
                batch_size=256,
                stride=1,
                horizon=50,
                horizon_weight=0.0,
                name=name,
            )
            history = result.history
            payload = {
                "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "history": history,
                "best_epoch": result.best_epoch,
                "best_error": result.best_error,
                "parameters": n_params(model),
            }
            torch.save(payload, path)
            model.to(device)
            model.eval()
            summary.setdefault("models", {})[name] = {
                "parameters": n_params(model),
                "best_epoch": result.best_epoch,
                "best_error": result.best_error,
                "history": history,
            }
            _history_path(folder).write_text(json.dumps(summary, indent=2))
        models[name] = model
        histories[name] = history
    return models, histories
