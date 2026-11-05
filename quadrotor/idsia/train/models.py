"""The three leftover families we train.

- ``phys_quotient``: frozen Crazyflie RK4 + body residual on the quotient
- ``phys_quotient_sym``: same RK4 + residual, Reynolds-averaged over Z₂×Z₂
- ``sym_quad_dynamics``: quotient + Z₂×Z₂, rigid-body integrate (no analytic Crazyflie model)

The MLP always writes body ``(Δp, Δv, Δθ, Δω)``. Polar parts lift through ``R``.
Published phys+res stays a loaded baseline in ``baselines/``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quadrotor.geometry import _so3_exp
from quadrotor.idsia.baselines.published import BaseQuadModel, PhysQuadModel
from quadrotor.symmetry import Orbit, Quotient, apply_body_dx, quotient, symmetrize_feature_stats

HIDDEN = 64
N_HIDDEN = 5


class NaiveResidual(nn.Module):
    """Geometry-free MLP ``ℝ^{in} → ℝ^{12}`` writing body ``(Δp, Δv, Δθ, Δω)``."""

    def __init__(self, in_dim: int, hidden: int = HIDDEN, n_hidden: int = N_HIDDEN) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        layers: list[nn.Module] = []
        dim = self.in_dim
        for _ in range(int(n_hidden)):
            layers += [nn.Linear(dim, hidden), nn.ReLU()]
            dim = int(hidden)
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dim, 12)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.out(self.mlp(features))


class BodyLeftover(BaseQuadModel):
    """MLP leftover in the body. Subclasses choose how to integrate it."""

    def __init__(self, residual: nn.Module, features: nn.Module) -> None:
        super().__init__()
        self.neural = residual
        self.feature_map = features
        dim = int(getattr(features, "dim"))
        self.register_buffer("feat_mean", torch.zeros(dim))
        self.register_buffer("feat_scale", torch.ones(dim))

    def set_feature_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if isinstance(self.neural, Orbit):
            # The orbit mirrors standardized features, so the standardization
            # must be mirror-invariant for exact equivariance.
            mean, scale = symmetrize_feature_stats(mean, scale)
        with torch.no_grad():
            self.feat_mean.copy_(mean.reshape_as(self.feat_mean).to(self.feat_mean))
            self.feat_scale.copy_(scale.reshape_as(self.feat_scale).to(self.feat_scale).clamp_min(1e-6))

    def body_dx(self, x: torch.Tensor, u: torch.Tensor, rotation: torch.Tensor | None = None) -> torch.Tensor:
        feat = (self.feature_map(x, u, rotation) - self.feat_mean) / self.feat_scale
        return self.neural(feat)


class NaiveBodyStep(BodyLeftover):
    """Integrate the body increment from the current state. No analytic physics."""

    def one_step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        rotation = _so3_exp(x[:, 6:9])  # once per step, shared by features and lift
        return apply_body_dx(x, x, self.body_dx(x, u, rotation), rotation)


class PhysicsResidual(BodyLeftover):
    """Frozen Crazyflie RK4, then add the body increment."""

    def __init__(self, residual: nn.Module, features: nn.Module) -> None:
        super().__init__(residual, features)
        self.phys = PhysQuadModel(zero_omega_dot=True)

    def one_step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_phys = self.phys.one_step(x, u)
        rotation = _so3_exp(x[:, 6:9])  # once per step, shared by features and lift
        return apply_body_dx(x_phys, x, self.body_dx(x, u, rotation), rotation)


class PhysQuotient(PhysicsResidual):
    """Physics + body residual on the 13-D quotient (no orbit)."""

    family = "phys_quotient"

    def __init__(self, hidden: int = HIDDEN, n_hidden: int = N_HIDDEN) -> None:
        super().__init__(NaiveResidual(13, hidden, n_hidden), Quotient())


class PhysQuotientSym(PhysicsResidual):
    """Physics + orbit-averaged body residual on the 13-D quotient."""

    family = "phys_quotient_sym"

    def __init__(self, hidden: int = HIDDEN, n_hidden: int = N_HIDDEN) -> None:
        super().__init__(Orbit(NaiveResidual(13, hidden, n_hidden)), Quotient())


class SymmetryQuadDynamicsNet(NaiveBodyStep):
    """Quotient + Z₂×Z₂ leftover, rigid-body integrate. No Crazyflie model."""

    family = "sym_quad_dynamics"

    def __init__(self, hidden: int = HIDDEN, n_hidden: int = N_HIDDEN) -> None:
        super().__init__(Orbit(NaiveResidual(13, hidden, n_hidden)), Quotient())


def make_model(family: str, symmetric: bool = True) -> nn.Module:
    if family == "phys_quotient":
        return PhysQuotient()
    if family == "phys_quotient_sym":
        return PhysQuotientSym() if symmetric else PhysQuotient()
    if family in {"sym_quad_dynamics", "quad_dynamics"}:
        return SymmetryQuadDynamicsNet()
    raise ValueError(f"unknown family {family}")


__all__ = [
    "NaiveResidual",
    "NaiveBodyStep",
    "PhysicsResidual",
    "PhysQuotient",
    "PhysQuotientSym",
    "SymmetryQuadDynamicsNet",
    "make_model",
    "quotient",
]
