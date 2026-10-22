# Quadrotor scripts

Run from the repo root, e.g. `.venv/bin/python -u quadrotor/scripts/idsia_mlp_hist10.py`.

**IDSIA**

- `idsia_mlp_hist10.py` — hist-10 Z₂×Z₂ quotient MLP vs physics / phys+res
- `idsia_mlp_hist5w.py` — hist-5 Z₂×Z₂ with stacked ω
- `idsia_melon_seeds.py` — 5-seed Melon table, train stride 5 (`runs/idsia_melon_seeds_s5/`)
- `idsia_hist10_stride5.py` — one Z₂×Z₂ vs headline

**NeuroBEM**

- `train_neurobem.py` — ParityMotorNet-24 + paper TCN
- `neurobem_gru_vs_coast.py` — official-test GRU vs coast
- `neurobem_nyu_openloop.py` — Rao et al. 100 Hz / 0.6 s coupled (open-loop) metrics
- `neurobem_mlp_fullstate.py` — shared 64-64 SE(3) trainer
- `neurobem_mlp_fullstate_parity200.py` — stateless parity, train h=200
- `neurobem_mlp_fullstate_parity200_hist.py` — hist-20 warm then predicted
- `neurobem_mlp_memory.py` — one-step history / product R²
