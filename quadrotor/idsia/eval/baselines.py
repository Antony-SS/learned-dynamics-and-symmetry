"""Melon MAE for the loaded baselines: analytic physics and the published phys+res checkpoint."""

from __future__ import annotations

import argparse

import torch

from quadrotor.evaluate import benchmark_mae
from quadrotor.geometry import train_device
from quadrotor.idsia.baselines import load_phys_res, load_physics
from quadrotor.idsia.data import load_idsia_prepared
from quadrotor.idsia.eval.evaluate import pack_mae, save_eval
from quadrotor.idsia.utils.idsia_training_reproduction_params import HORIZON, KEYS, PHYS_RES, PHYSICS


def _fmt(row: dict[str, list[float]]) -> str:
    parts = [" / ".join(f"{v:.4f}" for v in row[key]) for key in KEYS]
    return "  ".join(f"{c:28s}" for c in parts)


def load_baseline(name: str, device: str | torch.device = "cpu") -> torch.nn.Module:
    """``"physics"`` or ``"phys_res"`` (the authors' released checkpoint)."""
    if name == PHYSICS:
        return load_physics(device)
    if name == PHYS_RES:
        return load_phys_res(device=device)
    raise ValueError(f"unknown baseline {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="auto (MPS if available), cpu, mps, cuda")
    args = parser.parse_args()
    device = train_device(args.device)
    _train, _val, test = load_idsia_prepared()
    for name in (PHYSICS, PHYS_RES):
        mae = benchmark_mae(load_baseline(name, device), test, horizon=HORIZON, start_stride=1)
        save_eval(name, 0, mae)
        print(f"{name:16s}  {_fmt(pack_mae(mae))}", flush=True)


if __name__ == "__main__":
    main()
