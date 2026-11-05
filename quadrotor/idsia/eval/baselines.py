"""Melon MAE for published physics and phys+res. Does not train."""

from __future__ import annotations

import argparse

from quadrotor.geometry import train_device
from quadrotor.idsia.baselines import load_phys_res, load_physics
from quadrotor.idsia.data import load_idsia_prepared
from quadrotor.idsia.eval.evaluate import pack_mae, save_eval
from quadrotor.idsia.utils.idsia_training_reproduction_params import HORIZON, KEYS
from quadrotor.evaluate import benchmark_mae


def _fmt(row: dict[str, list[float]]) -> str:
    parts = [" / ".join(f"{v:.4f}" for v in row[key]) for key in KEYS]
    return "  ".join(f"{c:28s}" for c in parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = train_device(args.device)
    _train, _val, test = load_idsia_prepared()
    phys = load_physics(device)
    mae_p = benchmark_mae(phys, test, horizon=HORIZON, start_stride=1)
    save_eval("physics", 0, mae_p)
    print(f"{'physics':16s}  {_fmt(pack_mae(mae_p))}", flush=True)
    pub = load_phys_res(device=device)
    mae_r = benchmark_mae(pub, test, horizon=HORIZON, start_stride=1)
    save_eval("phys_res", 0, mae_r)
    print(f"{'phys+res':16s}  {_fmt(pack_mae(mae_r))}", flush=True)


if __name__ == "__main__":
    main()
