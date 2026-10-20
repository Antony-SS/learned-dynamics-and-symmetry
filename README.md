# Learning dynamics

Supervised learning of raw robot dynamics. The network predicts the observed step. It does not add a correction to a unicycle model or a rigid-body model. The experimental knob is how the input is canonicalized.

## Layout

- `src/learned_dynamics`: MLP and the full-batch trainer. The saved epoch is the one with the lowest held-out open-loop position error.
- `unicycle`: Robotarium ground robots. Reads the existing logs in `../robotarium-capstone-data` and does not copy them.
- `quadrotor`: Crazyflie 2.1 Brushless flights from the [IDSIA nano-drone system-identification benchmark](https://github.com/idsia-robotics/nanodrone-sysid-benchmark). Data stay under gitignored `data/`.

## Unicycle

Each training step sums three raw encoder steps, starting at the first pose of the log. The input is that sum, `(v·dt, ω·dt)` over `0.099 s`. The label is `Log(T_t^{-1} T_{t+3}) = (δx, δy, δθ)`. An SE(2) motion of the arena does not change the input.

Training uses robot 11. The second half of each diagonal lawnmower, the second half of the forward vertical lawnmower, and the last quarter of the forward spiral are held out. Training keeps three quarters of that spiral. The reverse spiral is excluded. The reverse vertical lawnmower and both horizontal lawnmowers stay in training. Open-loop rollout starts at the first pose of each segment. The full model integrates `T ← T ∘ Exp(f(v, ω))`. The residual model integrates `T ← T ∘ Z ∘ Exp(r(v, ω))`, where `Z = Exp(v·dt, 0, ω·dt)` is the wheel unicycle and `r` is the learned leftover. Wheel measurements stay on the log.

That label already contains the wheel motion: its forward part lines up with `v·dt`. The predicted step is initialized to the wheel twist `(v·dt, 0, ω·dt)`, and the network learns the rest of the same tangent vector. Training minimizes open-loop position error. There is no separate unicycle pose in the rollout.

## Quadrotor

The quadrotor fit uses the IDSIA benchmark's official 50-step protocol. Square, Random, and Chirp runs 1–3 train the models, run 4 validates them, and all Melon flights are test-only. Every window is seeded once from motion capture, then a GRU predicts body linear and angular acceleration from its rolled state and measured motor-speed history. Velocity and angular velocity are integrated from acceleration; position and SO(3) attitude are integrated from those velocities. No logged state is injected after the seed. Input normalization is fit on the training flights only.

- `absolute`: predicted world position, velocity, SO(3) rotation vector, body angular velocity, and four motor commands. In particular, xyz is the model's rolled-forward xyz, not another logged pose.
- `canonical`: body gravity direction, body velocity, body angular velocity, and the motors. It drops absolute position and yaw without using Euler angles.
- `symmetric`: the same canonical state, averaged over the left-right and front-back mirrors of the Crazyflie X layout.

Every network outputs only body-frame linear and angular acceleration. The physically coherent integrator determines all other state changes. The checkpoint is the lowest validation full-state MSE under uniform supervision across the complete 0.5-second horizon.

## Run

From this directory, in an environment with PyTorch:

```bash
python quadrotor/fetch_idsia.py
```

Open `unicycle/learn_dynamics.ipynb` or `quadrotor/learn_dynamics.ipynb`. The model classes sit in one cell. Inputs and outputs stay in physical units. Both notebooks score a fit by held-out open-loop position error.

The benchmark is from Busetto et al., *Control Engineering Practice* 172 (2026), 106871.
