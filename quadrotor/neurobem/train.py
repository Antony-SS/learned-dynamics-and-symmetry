"""Train NeuroBEM symmetric ParityMotorNet and the paper TCN at native 400 Hz."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quadrotor.neurobem.baselines import load_or_train_tcn
from quadrotor.neurobem.models import load_or_train


def main() -> None:
    print("NeuroBEM 400 Hz, 200-step (0.50 s) rollouts", flush=True)
    (train, val, test), model, history = load_or_train(device="mps", epochs=20)
    print(
        f"symmetric ready  train {len(train)}  val {len(val)}  test {len(test)}  "
        f"history {len(history)}",
        flush=True,
    )
    tcn, tcn_history = load_or_train_tcn(device="mps", epochs=20)
    print(f"tcn ready  history {len(tcn_history)}", flush=True)


if __name__ == "__main__":
    main()
