# Learning dynamics

Supervised learning of raw robot dynamics. The network predicts the observed step. The experimental knob is how the input is canonicalized.

## Layout

- `src/learned_dynamics`: MLP and the full-batch trainer. The saved epoch is the one with the lowest held-out open-loop position error.
- `unicycle`: SE(2) geometry and leftover wrappers. Robotarium experiments (`naive`, `quotient`, `quotient_aug`, `quotient_sym`, plus the analytic unicycle) live in `unicycle/experiments/`.
- `quadrotor/flight.py` and `quadrotor/geometry.py`: `PreparedFlight` and SO(3) / kinematics. `quadrotor/evaluate.py` is the open-loop MAE helper.
- `quadrotor/idsia`: Busetto et al. Crazyflie sys-id. Three leftovers we train plus their published physics / phys+res.
- Data stay under gitignored `data/`. Old one-off checkpoints remain in gitignored `runs/`.

## Run

From this directory, in an environment with PyTorch:

```bash
python -m quadrotor.idsia.data.fetch
python -m quadrotor.idsia.train.sym_quad_dynamics --seed 0
```

Open `unicycle/experiments/train/train.ipynb` or `quadrotor/idsia/train/train.ipynb`.

The IDSIA benchmark is from Busetto et al., *Control Engineering Practice* 172 (2026), 106871.
