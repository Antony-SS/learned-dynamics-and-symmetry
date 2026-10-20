"""Our IDSIA GRUs: naive, canonical, symmetric, even-odd."""

from __future__ import annotations

from pathlib import Path

import torch

from quadrotor.canonical import absolute_dim, canonical_dim, output_dim
from quadrotor.sequence import EvenOddSplitNet, MotorNet

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "runs" / "idsia_symmetry"
INCLUDE_VOLTAGE = False
NAMES = ("naive", "canonical", "symmetric", "even_odd")


class NaiveModel(MotorNet):
    reads_pose = True

    def __init__(self) -> None:
        super().__init__(absolute_dim(INCLUDE_VOLTAGE), output_dim(), mirror=False)


class CanonicalModel(MotorNet):
    def __init__(self) -> None:
        super().__init__(canonical_dim(INCLUDE_VOLTAGE), output_dim(), mirror=False)


class SymmetricModel(MotorNet):
    def __init__(self) -> None:
        super().__init__(canonical_dim(INCLUDE_VOLTAGE), output_dim(), mirror=True)


class EvenOddModel(EvenOddSplitNet):
    def __init__(self) -> None:
        super().__init__(canonical_dim(INCLUDE_VOLTAGE))


def _empty_model(name: str) -> torch.nn.Module:
    if name == "naive":
        return NaiveModel()
    if name == "canonical":
        return CanonicalModel()
    if name == "symmetric":
        return SymmetricModel()
    if name == "even_odd":
        return EvenOddModel()
    raise KeyError(name)


def load_trained(
    folder: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, torch.nn.Module]:
    """Load the already-trained IDSIA GRUs. Does not train."""
    folder = CHECKPOINT_DIR if folder is None else Path(folder)
    missing = [name for name in NAMES if not (folder / f"{name}.pt").is_file()]
    if missing:
        raise FileNotFoundError(
            f"trained IDSIA networks not under {folder}: {missing}. "
            "This notebook loads existing checkpoints and does not retrain."
        )
    loaded: dict[str, torch.nn.Module] = {}
    for name in NAMES:
        model = _empty_model(name)
        state = torch.load(folder / f"{name}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        loaded[name] = model
    return loaded
