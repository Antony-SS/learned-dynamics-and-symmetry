"""Argparse helpers for train / test CLIs."""

from __future__ import annotations

import argparse

from quadrotor.geometry import train_device
from unicycle.experiments.data import load_split
from unicycle.experiments.eval.metrics import eval_model, family_summary, load_physics
from unicycle.experiments.models import PHYSICS
from unicycle.experiments.train.lib import list_seeds, load_trained, train_family


def _fmt(row: dict) -> str:
    parts = []
    for key in ("1s", "5s", "30s", "full"):
        parts.append(f"{row[key]['position_rmse']:.4f}/{row[key]['heading_rmse']:.3f}")
    return "  ".join(parts)


def train_main(family: str) -> None:
    parser = argparse.ArgumentParser(description=f"Train {family}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_family(family, args.seed, device=train_device(args.device))


def test_main(family: str) -> None:
    parser = argparse.ArgumentParser(description=f"Test-set xyyaw for {family}")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = train_device(args.device)
    _train, _val, test = load_split()
    if family == PHYSICS:
        model = load_physics(device)
        print(f"{'seed':6s}  pos/heading RMSE at 1s  5s  30s  full", flush=True)
        row = eval_model(model, test, family, 0, device=device)
        print(f"{'phys':<6s}  {_fmt(row)}", flush=True)
        return
    seeds = list_seeds(family) if args.all or args.seed is None else [args.seed]
    if args.seed is not None and not args.all:
        seeds = [args.seed]
    if not seeds:
        raise SystemExit(f"no checkpoints under runs/unicycle/{family}/")
    print(f"{'seed':6s}  pos/heading RMSE at 1s  5s  30s  full", flush=True)
    for seed in seeds:
        model = load_trained(family, seed, device=device)
        row = eval_model(model, test, family, seed, device=device)
        print(f"{seed:<6d}  {_fmt(row)}", flush=True)
    if len(seeds) > 1:
        family_summary(family)
        print(f"wrote runs/unicycle/eval/{family}/summary.json", flush=True)
