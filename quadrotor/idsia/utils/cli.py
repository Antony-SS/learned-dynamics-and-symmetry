"""Argparse helpers for train_*.py / test_*.py."""

from __future__ import annotations

import argparse

from quadrotor.geometry import train_device
from quadrotor.idsia.data.load import load_idsia_prepared
from quadrotor.idsia.eval.evaluate import eval_model, family_summary
from quadrotor.idsia.train.lib import list_seeds, load_trained, train_family
from quadrotor.idsia.utils.idsia_training_reproduction_params import KEYS


def _fmt(row: dict[str, list[float]]) -> str:
    parts = []
    for key in KEYS:
        parts.append(" / ".join(f"{v:.4f}" for v in row[key]))
    return "  ".join(f"{c:28s}" for c in parts)


def train_main(family: str) -> None:
    parser = argparse.ArgumentParser(description=f"Train {family} with the IDSIA recipe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto (MPS if available), cpu, mps, cuda")
    parser.add_argument("--no-symmetric", action="store_true", help="disable Z2×Z2 on phys_quotient_sym")
    args = parser.parse_args()
    device = train_device(args.device)
    train_family(family, args.seed, device=device, symmetric=not args.no_symmetric)


def test_main(family: str) -> None:
    parser = argparse.ArgumentParser(description=f"Melon MAE for {family}")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="auto", help="auto (MPS if available), cpu, mps, cuda")
    args = parser.parse_args()
    device = train_device(args.device)
    _train, _val, test = load_idsia_prepared()
    seeds = list_seeds(family) if args.all or args.seed is None else [args.seed]
    if args.seed is not None and not args.all:
        seeds = [args.seed]
    if not seeds:
        raise SystemExit(f"no checkpoints under out/runs/{family}/")
    print(f"{'seed':6s}  {'p':28s}  {'R':28s}  {'v':28s}  {'ω'}", flush=True)
    for seed in seeds:
        model = load_trained(family, seed, device=device)
        mae = eval_model(model, test, family, seed)
        print(f"{seed:<6d}  {_fmt(mae)}", flush=True)
    if len(seeds) > 1:
        family_summary(family)
        print(f"wrote out/eval/{family}/summary.json", flush=True)
