# Learning dynamics

Supervised learning of raw robot dynamics. The network predicts the observed step. The experimental knob is how the input is canonicalized.

## Layout

- `src/learned_dynamics`: MLP and the full-batch trainer. The saved epoch is the one with the lowest held-out open-loop position error.
- `unicycle`: Robotarium ground robots. Notebooks in this folder; CLI trainers in `unicycle/scripts/`.
- `quadrotor/canonical.py` and `quadrotor/geometry.py`: `PreparedFlight` and SO(3) / kinematics. `quadrotor/evaluate.py` is the open-loop MAE helper.
- `quadrotor/idsia`: Busetto et al. Crazyflie sys-id. Three models we train (`QuadDynamicsNet`, `PhysQuotient`, `PhysQuotientSym`) plus their published physics / phys+res. CLIs and notebooks live in that folder.
- Data stay under gitignored `data/`. Old one-off checkpoints remain in gitignored `runs/`.

## Run

From this directory, in an environment with PyTorch:

```bash
python quadrotor/fetch_idsia.py
python -m quadrotor.idsia.train_quad_dynamics --seed 0
```

Open `unicycle/dynamics.ipynb` or `quadrotor/idsia/train.ipynb`.

The IDSIA benchmark is from Busetto et al., *Control Engineering Practice* 172 (2026), 106871.
