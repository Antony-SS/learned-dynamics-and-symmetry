"""Published IDSIA physics and phys+res loaders."""

from quadrotor.idsia.baselines.published import (
    BaseQuadModel,
    PhysQuadModel,
    PhysResQuadModel,
    ResidualQuadModel,
    fit_canonical_scalers,
    fit_scalers,
    fit_world_scalers,
    load_or_train_idsia_baselines,
    load_phys_res,
    load_physics,
    load_residual,
    load_theirs,
    train_phys_res,
)

__all__ = [
    "BaseQuadModel",
    "PhysQuadModel",
    "PhysResQuadModel",
    "ResidualQuadModel",
    "fit_canonical_scalers",
    "fit_scalers",
    "fit_world_scalers",
    "load_or_train_idsia_baselines",
    "load_phys_res",
    "load_physics",
    "load_residual",
    "load_theirs",
    "train_phys_res",
]
