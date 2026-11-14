# IDSIA Crazyflie sys-id

Busetto et al., *Control Engineering Practice* 172 (2026). Official split: Square / Random / Chirp runs 1–3 train, run 4 val, Melon test. Fetch with `python -m quadrotor.idsia.data.fetch`. Shared kinematics are `quadrotor.geometry`.

Layout: `data/` (clone + CSV loader), `baselines/` (their physics and phys+res models, plus the loader for the published checkpoint), `train/` (every family we train, including their phys+res retrained from scratch), `eval/` (Melon MAE), `utils/` (their training constants + CLI).

We train with **their recipe** (`utils/idsia_training_reproduction_params.py`: stride 1, 50-step windows, WeightedMSE λ=0.1, Adam 1e-5→1e-8 cosine T=500, 20 epochs, best val).

Every leftover we train writes the same **body** 12-D increment `(Δp, Δv, Δθ, Δω)`. Polar parts lift through current `R`; attitude composes as `R' = R Exp(Δθ)`. State for loss / RK4 is the 12-D chart `(p, v, Log R, ω)`.

## Ours vs. Theirs

Melon MAE at h = 1 / 10 / 50 on position, rotation, velocity and ω. Their analytic physics has no learned parameters and is used as published. Their physics+residual model is both loaded from the authors' released checkpoint and **retrained by us from scratch** (`phys_res_retrain`, seeds 0–4) with the same recipe and seeds as our families, so the Full-state comparison is like-for-like. We then keep their analytic RK4 and change only the leftover (quotient, then quotient + Z₂×Z₂). `SymmetryQuadDynamicsNet` is a separate test: it drops their vehicle model and relies on symmetry alone.

### Their baselines

They trained several networks. Only physics and physics+residual beat their own physics-only baseline on Melon, so those two are our comparison.

#### Pure physics

Analytic Crazyflie rigid body, RK4, `dt = 0.01`. Motors (rad/s) map to thrust and body torques (`kt`, `kc`, arm 3.53 cm, mass 45 g). Published step **holds ω fixed** (`ω̇ = 0`); position, velocity, and attitude still integrate under thrust.

- **Input:** world state `(p, v, Log R, ω)` and motors `Ω`.
- **Output:** next world state in the same 12-D chart.
- **Measures:** how far a first-principles vehicle model gets you with no learned leftover.



#### Physics + residual (their leftover)

A frozen copy of the physics above, plus an MLP on the **normalized world** chart. We use it two ways.

**Published (`phys_res`).** We load the authors' released checkpoint, `nanodrone-sysid-benchmark/out/models/phys+res_random_square_chirp.pt` (`baselines.load_phys_res`), and reproduce their reported Melon numbers exactly. It was saved at epoch 19 of a 500-epoch cosine schedule.

- **Input:** normalized `(p, v, Log R, ω)` and normalized `Ω` (16-D concat). Each channel is standardized with training-set statistics.
- **Output:** `x_phys + Δx` in the same normalized world chart, then denormalized.
- **Measures:** their leftover, exactly as published.

**Retrained (`phys_res_retrain`).** The same architecture (`PhysResQuadModel`: frozen RK4 + normalized-world MLP), trained from scratch with our recipe over seeds 0–4. This is the Full-state baseline in the paper. Train with `python -m quadrotor.idsia.train.phys_res_retrain --seed k` and score with `python -m quadrotor.idsia.eval.phys_res_retrain --all`.

### Our trained models

All four use the same body-leftover MLP (13 → 12, hidden 64 × 5). The first two keep their RK4 and change only the leftover. The last drops the vehicle model.

The two symmetric families standardize their features with mirror-invariant statistics (`symmetrize_feature_stats`), so the `Orbit` average is exactly equivariant. Checkpoints trained before this fix keep their saved statistics and are only approximately equivariant until retrained.

#### `PhysQuotient` (`phys_quotient`)

Their analytic RK4, with the leftover on the quotient and **no** morphological average. This is the first step away from their phys+res: the physics is the same, but the leftover no longer sees world `p` or yaw and no longer adds in the world chart.

- **Input:** 13-D quotient `(g_body, v_body, ω, Ω)`, one pass (no orbit).
- **Output:** body residual **added after** frozen RK4.
- **Measures:** does a body leftover on the quotient beat their world-chart phys+res, holding physics fixed?



#### `PhysQuotientAug` (`phys_quotient_aug`)

Exactly the `PhysQuotient` model, trained on mirror-augmented data: every training flight plus its three Z₂×Z₂ images (`quadrotor.symmetry.orbit_flights`). The mirror maps `p ↦ Mp`, `v ↦ Mv`, `R ↦ MRM`, `ω ↦ (det M) M ω`, and permutes rotors with the same tables `Orbit` uses. Validation flights and the loss scale stay unmirrored, so the objective is the same as `PhysQuotient`.

- **Input:** 13-D quotient, one pass (no orbit). 4× the training windows (~2.5 h per seed on MPS).
- **Output:** body residual after frozen RK4.
- **Measures:** symmetry learned from data (approximate) vs built into the architecture (`PhysQuotientSym`, exact).



#### `PhysQuotientSym` (`phys_quotient_sym`)

Same as `PhysQuotient`, plus Z₂×Z₂ Reynolds averaging on the residual.

- **Input:** 13-D quotient, four xz/yz mirrors.
- **Output:** orbit-averaged body residual after frozen RK4.
- **Measures:** does morphological symmetry still help once physics is already in the loop?



#### `SymmetryQuadDynamicsNet` (`sym_quad_dynamics`)

No Crazyflie force or torque model. The network predicts the whole step, and we only enforce rigid-body composition and airframe symmetry. It tests **symmetry as the only prior**.

- **Input:** 13-D quotient. `Orbit` evaluates all four mirrors and Reynolds-averages the body output.
- **Output:** body `(Δp, Δv, Δθ, Δω)`, integrated from the **current** state (`apply_body_dx` with no RK4).
- **Measures:** can quotient + Z₂×Z₂ match (or beat) their analytic physics **without** mass, inertia, or motor mixing?


| family              | class                     | prior                | input                  | output                       |
| ------------------- | ------------------------- | -------------------- | ---------------------- | ---------------------------- |
| `physics` (loaded)  | `PhysQuadModel`           | analytic RK4         | world 12-D + `Ω`       | next world 12-D              |
| `phys_res` (loaded) | `PhysResQuadModel`        | RK4 + world leftover | norm. world 12-D + `Ω` | world `x_phys + Δx`          |
| `phys_res_retrain`  | `PhysResQuadModel`        | RK4 + world leftover | norm. world 12-D + `Ω` | world `x_phys + Δx`          |
| `phys_quotient`     | `PhysQuotient`            | RK4 + leftover       | quotient 13-D          | body 12-D after RK4          |
| `phys_quotient_aug` | `PhysQuotientAug`         | RK4 + leftover       | quotient 13-D, mirror-augmented data | body 12-D after RK4 |
| `phys_quotient_sym` | `PhysQuotientSym`         | RK4 + leftover       | quotient 13-D, orbit   | body 12-D after RK4          |
| `sym_quad_dynamics` | `SymmetryQuadDynamicsNet` | rigid-body Exp only  | quotient 13-D, orbit   | body 12-D from current state |


```bash
python -m quadrotor.idsia.data.fetch
python -m quadrotor.idsia.train.phys_res_retrain --seed 0
python -m quadrotor.idsia.train.phys_quotient --seed 0
python -m quadrotor.idsia.train.phys_quotient_aug --seed 0
python -m quadrotor.idsia.eval.phys_res_retrain --all
python -m quadrotor.idsia.eval.phys_quotient --all
python -m quadrotor.idsia.eval.phys_quotient_aug --all
python -m quadrotor.idsia.eval.baselines
python -m quadrotor.idsia.visualizations.interactive
```

`train/train_quotient_all.sh` trains the retrained phys+res and every physics + quotient family over seeds 0–4, one run at a time (override with `FAMILIES`, `SEEDS`, `DEVICE`). `eval.baselines` scores physics and the published phys+res. `--all` evaluates every seed of a trained family that already has a checkpoint. Checkpoints: `runs/quadrotor/<family>/seed_k/{best.pt,last.pt}` (also `idsia/out/runs/`). Melon MAE: `idsia/out/eval/<family>/seed_k/{mae.json,predictions.npz}`.

`train/train.ipynb` trains one seed (resumes `last.pt`). `eval/results.ipynb` prints the Melon table at h=1 / 10 / 50. `visualizations/interactive` is a local Melon page: two models, start scrubber, gif window, 0.5 s channel plots.