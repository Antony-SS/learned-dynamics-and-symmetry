"""One-step leftover: frozen RK4 plus a body 12-D residual MLP.

Composition, not subclasses:

- ``quad_dynamics``: world 16-D features, no orbit
- ``phys_quotient``: 13-D quotient, no orbit
- ``phys_quotient_sym``: 13-D quotient, Z₂×Z₂ Reynolds orbit

The MLP always writes body ``(Δp, Δv, Δθ, Δω)``. Polar parts lift through ``R``.
Published phys+res stays a loaded baseline in ``baselines.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quadrotor.idsia.baselines import BaseQuadModel, PhysQuadModel
from quadrotor.symmetry import Orbit, Quotient, WorldFeatures, apply_body_dx, quotient

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


class PhysicsResidual(BaseQuadModel):
    """Frozen Crazyflie RK4, then add the residual's body increment."""

    def __init__(self, residual: nn.Module, features: nn.Module) -> None:
        super().__init__()
        self.phys = PhysQuadModel(zero_omega_dot=True)
        self.neural = residual
        self.feature_map = features
        dim = int(getattr(features, "dim"))
        self.register_buffer("feat_mean", torch.zeros(dim))
        self.register_buffer("feat_scale", torch.ones(dim))

    def set_feature_stats(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        with torch.no_grad():
            self.feat_mean.copy_(mean.reshape_as(self.feat_mean).to(self.feat_mean))
            self.feat_scale.copy_(scale.reshape_as(self.feat_scale).to(self.feat_scale).clamp_min(1e-6))

    def body_dx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        feat = (self.feature_map(x, u) - self.feat_mean) / self.feat_scale
        return self.neural(feat)

    def one_step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_phys = self.phys.one_step(x, u)
        return apply_body_dx(x_phys, x, self.body_dx(x, u))


class QuadDynamicsNet(PhysicsResidual):
    """Physics + body residual on world 16-D features (no orbit)."""

    family = "quad_dynamics"

    def __init__(self, symmetric: bool = False, hidden: int = HIDDEN, n_hidden: int = N_HIDDEN) -> None:
        del symmetric
        super().__init__(NaiveResidual(16, hidden, n_hidden), WorldFeatures())


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


def make_model(family: str, symmetric: bool = True) -> nn.Module:
    if family == "quad_dynamics":
        return QuadDynamicsNet()
    if family == "phys_quotient":
        return PhysQuotient()
    if family == "phys_quotient_sym":
        return PhysQuotientSym() if symmetric else PhysQuotient()
    raise ValueError(f"unknown family {family}")


__all__ = [
    "NaiveResidual",
    "PhysicsResidual",
    "QuadDynamicsNet",
    "PhysQuotient",
    "PhysQuotientSym",
    "make_model",
    "quotient",
]
