# Robotarium unicycle experiments

Commanded `(v, ω)` leftovers on robot 11 `standard_collections`. Shared SE(2) maps live in `unicycle/`.

Layout: `data.py` (split + mirrors), `models.py` (families), `train/` (5 s windows), `eval/` (test xyyaw), `utils/` (horizons + CLI), `visualizations/` (stills / gifs from the train notebook).

We train with **5 s** open-loop windows. The checkpoint is the lowest **5 s** xyyaw on the last **25%** of each training collection (same loss as train). That 25% is the val split. Test still reports 1 s / 5 s / 30 s / full.

State is `(x, y, yaw, v_body, ω_body)`. Commands are always commanded, never encoder-realized.

## Split

Train (first 75% / last 25% val): `arc_straight`, `figure8_vertical`, `lawnmower_horizontal`, `lawnmower_vertical`, `random_trajectory`, `spiral`, `spiral_cw`.

Held-out test (whole collections): `figure8`, `lawnmower_diagonal`, `slalom`.

## Physics baseline

Ideal unicycle on commanded `(v, ω)`: `Δx = v dt`, `Δy = 0`, `Δθ = ω dt`, then `apply_body_dx`. No parameters. Loaded, not trained.

- **Input:** current pose and `v_cmd, ω_cmd`.
- **Output:** next pose from the analytic plant.
- **Measures:** how far first-principles differential-drive kinematics get you.

## Trained families

Same leftover MLP (hidden 32 × 3). Output is body `(Δx, Δy, Δθ, Δv, Δω)`. Polar pose parts rotate with heading; the next rates are `rates + (Δv, Δω)`.

### `NaiveUnicycle` (`naive`)

Sees the arena pose.

- **Input:** 7-D `(x, y, θ, v, ω, v_cmd, ω_cmd)`.
- **Output:** 5-D body increment, rigid-body integrated from the current state.
- **Measures:** a geometry-free baseline.

### `QuotientUnicycleNet` (`quotient`)

Drops translation and heading.

- **Input:** 4-D `(v, ω, v_cmd, ω_cmd)`.
- **Output:** 5-D body increment.
- **Measures:** SE(2) quotient only.

### `QuotientUnicycleNet` + data augmentation (`quotient_aug`)

Same net as `quotient`. Training prefixes are concatenated with their left–right mirrors (`y, θ, ω, ω_cmd` flip). Val and test are not mirrored.

- **Measures:** does flipping the train set close the gap to an equivariant leftover?

### `QuotientSymUnicycleNet` (`quotient_sym`)

Quotient leftover, Reynolds-averaged over the two Z₂ elements (identity and left–right). Wheels swap; polar / axial signs on the leftover; v stays.

- **Input:** 4-D quotient; `Orbit` runs both group elements.
- **Output:** averaged 5-D body increment.
- **Measures:** hard morphological symmetry vs data-aug.

| family | class | prior | input | output |
| --- | --- | --- | --- | --- |
| (loaded) | `PhysicsUnicycle` | analytic unicycle | pose + command | next pose |
| `naive` | `NaiveUnicycle` | none | world 7-D | body 5-D |
| `quotient` | `QuotientUnicycleNet` | SE(2) quotient | 4-D rates+cmd | body 5-D |
| `quotient_aug` | same as quotient | quotient + mirrored train | 4-D | body 5-D |
| `quotient_sym` | `QuotientSymUnicycleNet` | quotient + Z₂ | 4-D, orbit | body 5-D |

```bash
python -m unicycle.experiments.train.naive --seed 0
python -m unicycle.experiments.eval.physics
python -m unicycle.experiments.eval.naive --all
python -m unicycle.experiments.interactive
```

`--all` evaluates every seed that already has a checkpoint. Checkpoints: `runs/unicycle/<family>/seed_k/{best.pt,last.pt}`. Test metrics: `runs/unicycle/eval/<family>/seed_k/test.json`.

`train/train.ipynb` starts with left/right counts, a commanded (v, ω) heatmap, then stills and gifs of every collection. It trains one seed (resumes `last.pt`). `eval/results.ipynb` prints test position / heading RMSE at 1 s, 5 s, 30 s, and the full log. `python -m unicycle.experiments.interactive` opens a local page: pick a `.pt` under `runs/unicycle/`, pick a collection, scrub the start time, play / stop the open-loop.
