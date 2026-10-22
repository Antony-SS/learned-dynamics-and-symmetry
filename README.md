# Learning dynamics

Supervised learning of raw robot dynamics. The network predicts the observed step. It does not add a correction to a unicycle model or a rigid-body model. The experimental knob is how the input is canonicalized.

## Layout

- `src/learned_dynamics`: MLP and the full-batch trainer. The saved epoch is the one with the lowest held-out open-loop position error.
- `unicycle`: Robotarium ground robots. Notebooks in this folder; CLI trainers in `unicycle/scripts/`.
- `quadrotor`: Crazyflie dynamics. Each benchmark is its own folder: `quadrotor/idsia` (Busetto et al.), `quadrotor/nanobench`, `quadrotor/neurobem`. Experiment CLIs live in `quadrotor/scripts/`. Data stay under gitignored `data/`.

## Run

From this directory, in an environment with PyTorch:

```bash
python quadrotor/fetch_idsia.py
```

Open `unicycle/dynamics.ipynb` or `quadrotor/idsia/idsia.ipynb`. Trainers are `unicycle/scripts/` and `quadrotor/scripts/`.

The benchmark is from Busetto et al., *Control Engineering Practice* 172 (2026), 106871.
