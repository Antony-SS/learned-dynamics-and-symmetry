"""Official-test ω (and pose) for the running NeuroBEM GRU vs coast."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from quadrotor.evaluate import benchmark_mae
from quadrotor.fit import SymmetricModel, unpack_state
from quadrotor.neurobem.data import HORIZON_STEPS, load_neurobem_prepared

CKPT = Path(__file__).resolve().parent / "out" / "parity18k" / "symmetric.pt"
HORIZONS = (1, 10, 50, 100, 200)


def coast_mae(flights, horizon: int = HORIZON_STEPS, stride: int = 200) -> dict[str, np.ndarray]:
    totals = {k: np.zeros(horizon) for k in ("position", "velocity", "rotation", "omega")}
    count = 0
    for flight in flights:
        last = flight.n_steps - horizon
        if last < 1:
            continue
        for start in range(0, last + 1, stride):
            p0 = flight.position[start]
            v0 = flight.velocity_world[start]
            w0 = flight.omega_body[start]
            r0 = flight.rotation[start]
            for k in range(horizon):
                totals["position"][k] += float(np.linalg.norm(flight.position[start + k + 1] - p0))
                totals["velocity"][k] += float(np.linalg.norm(flight.velocity_world[start + k + 1] - v0))
                rel = r0.T @ flight.rotation[start + k + 1]
                c = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
                totals["rotation"][k] += float(abs(np.arccos(c)))
                totals["omega"][k] += float(np.linalg.norm(flight.omega_body[start + k + 1] - w0))
            count += 1
    n = max(count, 1)
    return {k: v / n for k, v in totals.items()}


def main() -> None:
    train, val, test = load_neurobem_prepared()
    print(f"checkpoint {CKPT}  exists {CKPT.is_file()}", flush=True)
    if not CKPT.is_file():
        raise SystemExit("no GRU checkpoint yet")
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = SymmetricModel(include_voltage=False)
    model.load_state_dict(unpack_state(payload))
    model.eval()
    history = payload.get("history", []) if isinstance(payload, dict) else []
    if history:
        best = payload.get("best_epoch", history[-1].get("epoch"))
        print(
            f"loaded  best_epoch {best}  last val mse {history[-1].get('val_loss')}  "
            f"epochs logged {len(history)}",
            flush=True,
        )
    print("scoring official test, stride 200, warm 20", flush=True)
    mae = benchmark_mae(
        model, test, horizon=HORIZON_STEPS, start_stride=200, warm_steps=20, batch_size=64
    )
    coast = coast_mae(test)
    print(f"{'h':>4} {'t':>7}  {'coast ω':>8} {'GRU ω':>8}  {'coast p':>8} {'GRU p':>8}  {'GRU R':>8}")
    for h in HORIZONS:
        print(
            f"{h:4d} {1000.0 * h / 400:6.1f}ms  "
            f"{coast['omega'][h - 1]:8.4f} {mae['omega'][h - 1]:8.4f}  "
            f"{coast['position'][h - 1]:8.4f} {mae['position'][h - 1]:8.4f}  "
            f"{mae['rotation'][h - 1]:8.4f}"
        )


if __name__ == "__main__":
    main()
