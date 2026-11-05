"""Commanded body-step families. Learned output is (Δx, Δy, Δθ, Δv, Δω)."""

from __future__ import annotations

import torch
import torch.nn as nn

from unicycle.symmetry import (
    SymmetricQuotientMLP,
    Quotient,
    WorldFeatures,
    apply_body_dx,
    apply_pose_dx,
    physics_body_dx,
    symmetrize_feature_stats,
)

HIDDEN = 32
N_HIDDEN = 3
# Legacy 4-D heads wrote ``(Δx, Δθ, Δv, Δω)``; current ``BodyMlp`` is five with Δy at index 1.
_LEGACY_OUT_ROWS = (0, 2, 3, 4)

NAIVE = "naive"
QUOTIENT = "quotient"
QUOTIENT_AUG = "quotient_aug"
QUOTIENT_SYM = "quotient_sym"
PHYSICS = "physics"
TRAIN_FAMILIES = (NAIVE, QUOTIENT, QUOTIENT_AUG, QUOTIENT_SYM)
FAMILIES = (PHYSICS,) + TRAIN_FAMILIES


class BodyMlp(nn.Module):
    """Geometry-free MLP. Writes packed ``(Δx, Δy, Δθ, Δv, Δω)``."""

    def __init__(self, in_dim: int, hidden: int = HIDDEN, n_hidden: int = N_HIDDEN) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(in_dim)
        for _ in range(int(n_hidden)):
            layers += [nn.Linear(dim, hidden), nn.ReLU()]
            dim = int(hidden)
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dim, 5)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.out(self.mlp(features))


class BodyStep(nn.Module):
    def __init__(self, net: nn.Module, features: nn.Module) -> None:
        super().__init__()
        self.neural = net
        self.feature_map = features
        dim = int(getattr(features, "dim"))
        self.register_buffer("feat_mean", torch.zeros(dim))
        self.register_buffer("feat_scale", torch.ones(dim))

    def set_feature_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if isinstance(self.neural, SymmetricQuotientMLP):
            # The group average mirrors normalized features; keep the mirror at raw ω = 0.
            mean, scale = symmetrize_feature_stats(mean, scale)
        with torch.no_grad():
            self.feat_mean.copy_(mean.reshape_as(self.feat_mean).to(self.feat_mean))
            self.feat_scale.copy_(scale.reshape_as(self.feat_scale).to(self.feat_scale).clamp_min(1e-6))

    def body_dx(self, pose: torch.Tensor, rates: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        feat = (self.feature_map(pose, rates, command) - self.feat_mean) / self.feat_scale
        return self.neural(feat)

    def step(
        self,
        pose: torch.Tensor,
        rates: torch.Tensor,
        command: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del dt
        return apply_body_dx(pose, rates, self.body_dx(pose, rates, command))

    def rollout(self, pose: torch.Tensor, rates: torch.Tensor, commands: torch.Tensor, dt: float) -> torch.Tensor:
        """Open-loop poses after each command. ``commands`` is (..., T, 2)."""
        preds = []
        for t in range(commands.shape[-2]):
            pose, rates = self.step(pose, rates, commands[..., t, :], dt)
            preds.append(pose)
        return torch.stack(preds, dim=-2)


class PhysicsUnicycle(nn.Module):
    """Ideal unicycle on commanded (v, ω). No parameters."""

    def body_dx(self, pose: torch.Tensor, rates: torch.Tensor, command: torch.Tensor, dt: float) -> torch.Tensor:
        del pose, rates
        return physics_body_dx(command, dt)

    def step(
        self,
        pose: torch.Tensor,
        rates: torch.Tensor,
        command: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del rates
        dx = physics_body_dx(command, dt)
        return apply_pose_dx(pose, dx), command

    def rollout(self, pose: torch.Tensor, rates: torch.Tensor, commands: torch.Tensor, dt: float) -> torch.Tensor:
        preds = []
        for t in range(commands.shape[-2]):
            pose, rates = self.step(pose, rates, commands[..., t, :], dt)
            preds.append(pose)
        return torch.stack(preds, dim=-2)


def _remap_out_head(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Align ``BodyMlp`` output layers between legacy 4-D and current 5-D checkpoints."""
    in_features = weight.shape[1]
    ckpt_out = weight.shape[0]
    if ckpt_out == out_features:
        return weight, bias
    if ckpt_out == 5 and out_features == 4:
        rows = weight.new_tensor(_LEGACY_OUT_ROWS, dtype=torch.long)
        new_bias = None if bias is None else bias.index_select(0, rows)
        return weight.index_select(0, rows), new_bias
    if ckpt_out == 4 and out_features == 5:
        new_w = weight.new_zeros(5, in_features)
        new_w[0] = weight[0]
        new_w[2] = weight[1]
        new_w[3] = weight[2]
        new_w[4] = weight[3]
        new_b = None
        if bias is not None:
            new_b = bias.new_zeros(5)
            new_b[0] = bias[0]
            new_b[2] = bias[1]
            new_b[3] = bias[2]
            new_b[4] = bias[3]
        return new_w, new_b
    raise ValueError(f"cannot remap MLP out head {ckpt_out} → {out_features}")


def adapt_checkpoint_state(
    state: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Remap ``BodyMlp`` output layers when checkpoint and model head sizes differ."""
    targets = model.state_dict()
    adapted = dict(state)
    for key, weight in state.items():
        if not key.endswith(".out.weight"):
            continue
        target = targets.get(key)
        if target is None or weight.shape == target.shape:
            continue
        bias_key = key.replace(".weight", ".bias")
        ckpt_bias = state.get(bias_key)
        new_w, new_b = _remap_out_head(
            weight,
            ckpt_bias,
            out_features=int(target.shape[0]),
        )
        adapted[key] = new_w
        if new_b is not None and bias_key in targets:
            adapted[bias_key] = new_b
    return adapted


def load_state(model: nn.Module, payload: dict | dict[str, torch.Tensor]) -> None:
    state = payload["state"] if isinstance(payload, dict) and "state" in payload else payload
    adapted = adapt_checkpoint_state(state, model)
    targets = model.state_dict()
    compatible = {
        key: tensor
        for key, tensor in adapted.items()
        if key in targets and tensor.shape == targets[key].shape
    }
    model.load_state_dict(compatible, strict=False)


def make_model(family: str) -> nn.Module:
    if family == PHYSICS:
        return PhysicsUnicycle()
    if family == NAIVE:
        return BodyStep(BodyMlp(WorldFeatures.dim), WorldFeatures())
    if family in (QUOTIENT, QUOTIENT_AUG):
        return BodyStep(BodyMlp(Quotient.dim), Quotient())
    if family == QUOTIENT_SYM:
        return BodyStep(SymmetricQuotientMLP(BodyMlp(Quotient.dim)), Quotient())
    raise ValueError(f"unknown family {family}")
