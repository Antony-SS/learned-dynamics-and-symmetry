# IDSIA Crazyflie sys-id

Busetto et al., *Control Engineering Practice* 172 (2026). Official split: Square / Random / Chirp runs 1–3 train, run 4 val, Melon test. Fetch with `python -m quadrotor.idsia.data.fetch`. Shared kinematics are `quadrotor.geometry`.

Layout: `data/` (clone + CSV loader), `baselines/` (published physics / phys+res), `train/` (our leftover families), `eval/` (Melon MAE), `utils/` (their training constants + CLI).

We train with **their recipe** (`utils/idsia_training_reproduction_params.py`: stride 1, 50-step windows, WeightedMSE λ=0.1, Adam 1e-5→1e-8 cosine T=500, 20 epochs, best val).

Every leftover we train writes the same **body** 12-D increment `(Δp, Δv, Δθ, Δω)`. Polar parts lift through current `R`; attitude composes as `R' = R Exp(Δθ)`. State for loss / RK4 is the 12-D chart `(p, v, Log R, ω)`.

## Ours vs. Theirs

Melon MAE at h = 1 / 10 / 50 on position, rotation, velocity, ω. Analytic physics is the only published/loaded baseline. Their phys+res architecture we retrain ourselves (`runs/quadrotor/phys_res/`, 5 seeds) — the official checkpoint was undertrained. We then keep their analytic RK4 and change only the leftover (quotient, then quotient + Z₂×Z₂). `SymmetryQuadDynamicsNet` is a separate test: drop their vehicle model and rely on symmetry alone.

### Their baselines

They trained several nets; only physics and physics+residual beat “just physics” on Melon, so those two are the comparison.

#### Pure physics

Analytic Crazyflie rigid body, RK4, `dt = 0.01`. Motors (rad/s) map to thrust and body torques (`kt`, `kc`, arm 3.53 cm, mass 45 g). Published step **holds ω fixed** (`ω̇ = 0`); position, velocity, and attitude still integrate under thrust.

- **Input:** world state `(p, v, Log R, ω)` and motors `Ω`.
- **Output:** next world state in the same 12-D chart.
- **Measures:** how far a first-principles vehicle model gets you with no learned leftover.



#### Physics + residual (their leftover)

Frozen copy of the physics above, plus an MLP on the **normalized world** chart.  We reproduced their published numbers, and did slightly better (theirs was actually slightly undertrained).  

- **Input:** normalized `(p, v, Log R, ω)` and normalized `Ω` (16-D concat) where normalization is done per channel on the training data.  This is standard.
- **Output:** `x_phys + Δx` in the same normalized world chart, then denormalized.
- **Measures:** their leftover, trained with their recipe (5 seeds). Table uses `runs/quadrotor/phys_res/` which are our trained reproductions, which fair slightly better than their published`phys+res_random_square_chirp.pt`.

### Our trained models

Same body leftover MLP (13 → 12, hidden 64 × 5). First two keep their RK4 and only change the leftover. The last drops the vehicle model.

#### `PhysQuotient` (`phys_quotient`)

Their analytic RK4, leftover on the quotient, **no** morphological average. First step off their phys+res: same physics, leftover no longer sees world `p` / yaw and no longer adds in the world chart.

- **Input:** 13-D quotient `(g_body, v_body, ω, Ω)`, one pass (no orbit).
- **Output:** body residual **added after** frozen RK4.
- **Measures:** does a body leftover on the quotient beat their world-chart phys+res, holding physics fixed?



#### `PhysQuotientSym` (`phys_quotient_sym`)

Same as `PhysQuotient`, plus Z₂×Z₂ Reynolds averaging on the residual.

- **Input:** 13-D quotient, four xz/yz mirrors.
- **Output:** orbit-averaged body residual after frozen RK4.
- **Measures:** does morphological symmetry still help once physics is already in the loop?



#### `SymmetryQuadDynamicsNet` (`sym_quad_dynamics`)

No Crazyflie force/torque model. The net predicts the whole step; we only enforce rigid-body composition and airframe symmetry. Interesting as a test of **symmetry as the only prior**.

- **Input:** 13-D quotient. `Orbit` evaluates all four mirrors and Reynolds-averages the body output.
- **Output:** body `(Δp, Δv, Δθ, Δω)`, integrated from the **current** state (`apply_body_dx` with no RK4).
- **Measures:** can quotient + Z₂×Z₂ match (or beat) their analytic physics **without** mass, inertia, or motor mixing?


| family              | class                     | prior                | input                  | output                       |
| ------------------- | ------------------------- | -------------------- | ---------------------- | ---------------------------- |
| (loaded)            | `PhysQuadModel`           | analytic RK4         | world 12-D + `Ω`       | next world 12-D              |
| `phys_res`          | `PhysResQuadModel`        | RK4 + world leftover | norm. world 12-D + `Ω` | world `x_phys + Δx`          |
| `phys_quotient`     | `PhysQuotient`            | RK4 + leftover       | quotient 13-D          | body 12-D after RK4          |
| `phys_quotient_sym` | `PhysQuotientSym`         | RK4 + leftover       | quotient 13-D, orbit   | body 12-D after RK4          |
| `sym_quad_dynamics` | `SymmetryQuadDynamicsNet` | rigid-body Exp only  | quotient 13-D, orbit   | body 12-D from current state |


```bash
python -m quadrotor.idsia.data.fetch
python -m quadrotor.idsia.train.phys_quotient --seed 0
python -m quadrotor.idsia.eval.phys_quotient --all
python -m quadrotor.idsia.eval.baselines
python -m quadrotor.idsia.visualizations.interactive
```

`--all` evaluates every seed that already has a checkpoint. Checkpoints: `runs/quadrotor/<family>/seed_k/{best.pt,last.pt}` (also `idsia/out/runs/`). Melon MAE: `idsia/out/eval/<family>/seed_k/{mae.json,predictions.npz}`.

`train/train.ipynb` trains one seed (resumes `last.pt`). `eval/results.ipynb` prints the Melon table at h=1 / 10 / 50. `visualizations/interactive` is a local Melon page: two models, start scrubber, gif window, 0.5 s channel plots.