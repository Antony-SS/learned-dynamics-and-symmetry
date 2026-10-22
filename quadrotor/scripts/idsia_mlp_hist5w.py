"""IDSIA hist-5 Z2×Z2 quotient MLP with stacked ω."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quadrotor.idsia.history_mlp import HIST5_DIR, run

if __name__ == "__main__":
    run(history=5, symmetric=True, out=HIST5_DIR, seed=0, tag="hist-5 Z2×Z2 +ω")
