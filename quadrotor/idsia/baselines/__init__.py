"""Published IDSIA baselines: analytic physics and the released physics+residual checkpoint."""

from quadrotor.idsia.baselines.published import (
    BaseQuadModel,
    PhysQuadModel,
    PhysResQuadModel,
    ResidualQuadModel,
    fit_canonical_scalers,
    fit_scalers,
    fit_world_scalers,
    load_phys_res,
    load_physics,
    load_residual,
)

__all__ = [
    "BaseQuadModel",
    "PhysQuadModel",
    "PhysResQuadModel",
    "ResidualQuadModel",
    "fit_canonical_scalers",
    "fit_scalers",
    "fit_world_scalers",
    "load_phys_res",
    "load_physics",
    "load_residual",
]
