"""Melon MAE for analytic physics and our phys+res reproductions."""

from __future__ import annotations

import argparse

from quadrotor.geometry import train_device
from quadrotor.idsia.baselines import load_physics
from quadrotor.idsia.data import load_idsia_prepared
from quadrotor.idsia.eval.evaluate import eval_model, family_summary, pack_mae, save_eval
from quadrotor.idsia.train.lib import list_seeds, load_trained
from quadrotor.idsia.utils.idsia_training_reproduction_params import HORIZON, KEYS, PHYS_RES
from quadrotor.evaluate import benchmark_mae


def _fmt(row: dict[str, list[float]]) -> str:
    parts = [" / ".join(f"{v:.4f}" for v in row[key]) for key in KEYS]
    return "  ".join(f"{c:28s}" for c in parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="auto (MPS if available), cpu, mps, cuda")
    args = parser.parse_args()
    device = train_device(args.device)
    _train, _val, test = load_idsia_prepared()
    phys = load_physics(device)
    mae_p = benchmark_mae(phys, test, horizon=HORIZON, start_stride=1)
    save_eval("physics", 0, mae_p)
    print(f"{'physics':16s}  {_fmt(pack_mae(mae_p))}", flush=True)
    seeds = list_seeds(PHYS_RES)
    if not seeds:
        raise SystemExit("no phys_res checkpoints under runs/quadrotor/phys_res/")
    for seed in seeds:
        model = load_trained(PHYS_RES, seed, device=device)
        row = eval_model(model, test, PHYS_RES, seed)
        print(f"{'phys_res s'+str(seed):16s}  {_fmt(row)}", flush=True)
    if len(seeds) > 1:
        summary = family_summary(PHYS_RES)
        mean = {k: summary[k]["mean"] for k in KEYS}
        print(f"{'phys_res mean':16s}  {_fmt(mean)}", flush=True)


if __name__ == "__main__":
    main()
