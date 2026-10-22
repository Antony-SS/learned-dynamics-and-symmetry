# Learning dynamics

Supervised learning of raw robot dynamics. The network predicts the observed step. It does not add a correction to a unicycle model or a rigid-body model. The experimental knob is how the input is canonicalized.

## Layout

- `src/learned_dynamics`: MLP and the full-batch trainer. The saved epoch is the one with the lowest held-out open-loop position error.
- `unicycle`: Robotarium ground robots. Notebooks in this folder; CLI trainers in `unicycle/scripts/`.
- `quadrotor`: Crazyflie dynamics. Each benchmark is its own folder: `quadrotor/idsia` (Busetto et al.), `quadrotor/nanobench`, `quadrotor/neurobem`. Experiment CLIs live in `quadrotor/scripts/`. Data stay under gitignored `data/`.

## Unicycle

Each training step sums three raw encoder steps, starting at the first pose of the log. The input is that sum, `(v·dt, ω·dt)` over `0.099 s`. The label is `Log(T_t^{-1} T_{t+3}) = (δx, δy, δθ)`. An SE(2) motion of the arena does not change the input.

Training uses robot 11. The second half of each diagonal lawnmower, the second half of the forward vertical lawnmower, and the last quarter of the forward spiral are held out. Training keeps three quarters of that spiral. The reverse spiral is excluded. The reverse vertical lawnmower and both horizontal lawnmowers stay in training. Open-loop rollout starts at the first pose of each segment. The full model integrates `T ← T ∘ Exp(f(v, ω))`. The residual model integrates `T ← T ∘ Z ∘ Exp(r(v, ω))`, where `Z = Exp(v·dt, 0, ω·dt)` is the wheel unicycle and `r` is the learned leftover. Wheel measurements stay on the log.

That label already contains the wheel motion: its forward part lines up with `v·dt`. The predicted step is initialized to the wheel twist `(v·dt, 0, ω·dt)`, and the network learns the rest of the same tangent vector. Training minimizes open-loop position error. There is no separate unicycle pose in the rollout.

## Quadrotor

The IDSIA leftover is a hist-10 64-64 MLP on the stacked quotient $(g^{\mathrm{body}}, v^{\mathrm{body}}, \omega, \Omega)$, orbit-averaged over both Crazyflie mirrors. Square / Random / Chirp runs 1–3 train, run 4 validates, Melon tests. Each window is seeded from motion capture, then the net predicts body $(a, \alpha)$ and the SE(3) integrator rolls pose. No GRU. Physics and physics+residual stay as the paper baselines.

## Run

From this directory, in an environment with PyTorch:

```bash
python quadrotor/fetch_idsia.py
```

Open `unicycle/dynamics.ipynb` or `quadrotor/idsia/idsia.ipynb`. Trainers are `unicycle/scripts/` and `quadrotor/scripts/`.

The benchmark is from Busetto et al., *Control Engineering Practice* 172 (2026), 106871.
