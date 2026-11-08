# Robotarium unicycle experiments

Commanded `(v, ω)` dynamics on robot 11 `standard_collections`. Shared SE(2) maps live in `unicycle/`.

Layout: `data.py` (split + mirrors), `models.py` (families), `train/` (open-loop windows), `eval/` (test xyyaw), `utils/` (horizons + CLI), `visualizations/` (stills / gifs from the train notebook).

We train with **10 s** open-loop windows (`HORIZON_S` in `utils/params.py`). The checkpoint is the lowest **10 s** xyyaw on the last **25%** of each training collection (same loss as train). That 25% is the val split. Test still reports 1 s / 10 s / 30 s / full.

State is `(x, y, yaw, v_body, ω_body)`. Commands are always commanded, never encoder-realized.

## Split

Train (first 75% / last 25% val): `arc_straight`, `figure8_vertical`, `lawnmower_horizontal`, `lawnmower_vertical`, `random_crawl`, `rounded_rects`, `rounded_rects_cw`, `spiral`, `spiral_cw`.

Held-out test (whole collections): `figure8`, `lawnmower_diagonal`, `random_trajectory`, `slalom`.

## Physics baseline

Ideal unicycle on commanded `(v, ω)`: `Δx = v dt`, `Δy = 0`, `Δθ = ω dt`, then `apply_body_dx`. No parameters. Loaded, not trained.

- **Input:** current pose and `v_cmd, ω_cmd`.
- **Output:** next pose from the analytic plant.
- **Measures:** how far first-principles differential-drive kinematics get you.

## Trained families

Same MLP (hidden 32 × 3). Output is `(Δx, Δy, Δθ, Δv, Δω)`: the learned models predict sideslip, no no-slip constraint. Body (Δx, Δy) rotates with heading; the next rates are `rates + (Δv, Δω)`. Checkpoints load strictly: a checkpoint whose shapes do not match the current model raises an error.

### Naive (`naive`)

Sees the arena pose.

- **Input:** 7-D `(x, y, θ, v, ω, v_cmd, ω_cmd)`.
- **Output:** `(Δx, 0, Δθ, Δv, Δω)`, rigid-body integrated from the current state.
- **Measures:** a geometry-free baseline.

### Quotient (`quotient`)

Drops translation and heading.

- **Input:** 4-D `(v, ω, v_cmd, ω_cmd)`.
- **Output:** `(Δx, 0, Δθ, Δv, Δω)`.
- **Measures:** SE(2) quotient only.

### Quotient + data augmentation (`quotient_aug`)

Same net as `quotient`. Training prefixes are concatenated with their left–right mirrors (`y, θ, ω, ω_cmd` flip). Val and test are not mirrored.

- **Measures:** does flipping the train set close the gap to an equivariant map?

### Symmetric quotient (`quotient_sym`)

The same quotient MLP, averaged over the two Z₂ elements (identity and the left–right mirror). The mirror swaps the wheels: v stays and ω flips, and the increment gets the matching polar and axial signs. Features are standardized with mirror-invariant statistics (zero mean on ω and ω_cmd), so the average is exactly equivariant.

- **Input:** 4-D quotient. `SymmetricQuotientMLP` runs both group elements in one batch.
- **Output:** averaged `(Δx, 0, Δθ, Δv, Δω)`.
- **Measures:** hard morphological symmetry vs data-aug.

| family | class | prior | input | output |
| --- | --- | --- | --- | --- |
| (loaded) | `PhysicsUnicycle` | analytic unicycle | pose + command | next pose |
| `naive` | `BodyStep(BodyMlp(7), WorldFeatures())` | none | world 7-D | Δx, Δy, Δθ, Δv, Δω |
| `quotient` | `BodyStep(BodyMlp(4), Quotient())` | SE(2) quotient | 4-D rates + cmd | Δx, Δy, Δθ, Δv, Δω |
| `quotient_aug` | same as `quotient` | quotient + mirrored train | 4-D | Δx, Δy, Δθ, Δv, Δω |
| `quotient_sym` | `BodyStep(SymmetricQuotientMLP(BodyMlp(4)), Quotient())` | quotient + Z₂ | 4-D, group average | Δx, Δy, Δθ, Δv, Δω |

```bash
python -m unicycle.experiments.train.naive --seed 0
python -m unicycle.experiments.eval.physics
python -m unicycle.experiments.eval.naive --all
python -m unicycle.experiments.interactive
```

`--all` evaluates every seed that already has a checkpoint. Checkpoints: `runs/unicycle/<family>/seed_k/{best.pt,last.pt}`. Test metrics: `runs/unicycle/eval/<family>/seed_k/test.json`.

`train/train.ipynb` starts with left/right counts, a commanded (v, ω) heatmap, then stills and gifs of every collection. It trains one seed (resumes `last.pt`). `eval/results.ipynb` prints test position / heading MAE at 1 s, 10 s, 30 s (error at that step, averaged over every start, as in the quad benchmark), and averaged along the full log. `python -m unicycle.experiments.interactive` opens a local page showing one checkpoint per family: the seed with the lowest test xyyaw at 10 s (seed 0 if no family has test results yet), on the four test collections only. Pick a collection, scrub the start time, play / stop the open-loop.
