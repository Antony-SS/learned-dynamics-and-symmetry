"""Commanded body-step families. Output is always (Δx, Δy, Δθ)."""

from __future__ import annotations

import torch
import torch.nn as nn

from unicycle.symmetry import (
    Orbit,
    Quotient,
    WorldFeatures,
    apply_body_dx,
    physics_body_dx,
)

HIDDEN = 32
N_HIDDEN = 3

NAIVE = "naive"
QUOTIENT = "quotient"
QUOTIENT_AUG = "quotient_aug"
QUOTIENT_SYM = "quotient_sym"
PHYSICS = "physics"
TRAIN_FAMILIES = (NAIVE, QUOTIENT, QUOTIENT_AUG, QUOTIENT_SYM)
FAMILIES = (PHYSICS,) + TRAIN_FAMILIES


class BodyMlp(nn.Module):
    """Geometry-free MLP. Writes the full body step (Δx, Δy, Δθ)."""

    def __init__(self, in_dim: int, hidden: int = HIDDEN, n_hidden: int = N_HIDDEN) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(in_dim)
        for _ in range(int(n_hidden)):
            layers += [nn.Linear(dim, hidden), nn.ReLU()]
            dim = int(hidden)
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dim, 3)
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
        dx = self.body_dx(pose, rates, command)
        next_pose = apply_body_dx(pose, dx)
        next_rates = torch.stack((dx[..., 0] / dt, dx[..., 2] / dt), dim=-1)
        return next_pose, next_rates

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
        return apply_body_dx(pose, dx), command

    def rollout(self, pose: torch.Tensor, rates: torch.Tensor, commands: torch.Tensor, dt: float) -> torch.Tensor:
        preds = []
        for t in range(commands.shape[-2]):
            pose, rates = self.step(pose, rates, commands[..., t, :], dt)
            preds.append(pose)
        return torch.stack(preds, dim=-2)


def make_model(family: str) -> nn.Module:
    if family == PHYSICS:
        return PhysicsUnicycle()
    if family == NAIVE:
        return BodyStep(BodyMlp(WorldFeatures.dim), WorldFeatures())
    if family in (QUOTIENT, QUOTIENT_AUG):
        return BodyStep(BodyMlp(Quotient.dim), Quotient())
    if family == QUOTIENT_SYM:
        return BodyStep(Orbit(BodyMlp(Quotient.dim)), Quotient())
    raise ValueError(f"unknown family {family}")
