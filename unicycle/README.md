# Unicycle leftovers

Shared SE(2) geometry and symmetry wrappers. Robotarium experiments live in `[experiments/](experiments/)`.

## Overview

`symmetry.py` wraps an arbitrary MLP. The net writes a **body** increment `(Δx, Δy, Δθ, Δv, Δω)`. We integrate that increment on SE(2): polar pose lifts through heading; rates add in the body. Commands are the Robotarium commanded `(v, ω)`, not encoder-realized rates.

Contract:

- `BodyMlp` — geometry-free MLP. Output is the full-state body increment `(Δx, Δy, Δθ, Δv, Δω)`.
- `Quotient` — drop arena pose. Features are `(v_body, ω_body, v_cmd, ω_cmd)`.
- `Orbit` — Reynolds average over the left–right mirror (Z₂).
- `apply_body_dx` — rigid-body compose of the 5-D increment:

```
x' = x + cosθ Δx − sinθ Δy
y' = y + sinθ Δx + cosθ Δy
θ' = wrap(θ + Δθ)
v' = v + Δv
ω' = ω + Δω
```

The analytic unicycle plant (physics baseline) uses `Δx = v_cmd dt`, `Δy = 0`, `Δθ = ω_cmd dt` and sets the next rates to the command. Learned families predict the increment on all five channels; they do not back out rates from `Δx / dt`.

## Math

Configuration is SE(2): position (x, y) and heading θ. The body twist is (v, ω). `Log` / `Exp` are the GTSAM Pose2 chart (ρx, ρy, φ). Training state is `(x, y, θ, v_body, ω_body)` with command `(v_cmd, ω_cmd)`.

A body increment is written in the robot frame. Polar (Δx, Δy) rotate with heading; Δθ, Δv, Δω stay in the body algebra.

## Continuous symmetries

The plane is invariant under translation and rotation of the arena. Shifting (x, y) or adding a constant to θ (and rotating the position) does not change body-relative motion.

The quotient drops (x, y, θ). What remains is the body rates and the command, `(v_body, ω_body, v_cmd, ω_cmd)`.

## Morphological symmetries

A differential-drive robot is unchanged under reflection through the body xz plane (left–right).

**Polar** (v, Δx, Δy, Δv): v, Δx, Δv stay; Δy flips. **Axial / pseudo** (ω, Δθ, Δω, ω_cmd): pick up det = −1. Wheel swap `(left, right)` → `(right, left)` is exactly “v stays, ω flips”.

`Orbit` stacks identity and the left–right image, runs the MLP once on that doubled batch, maps each 5-D leftover back with the matching polar / axial signs, and averages (Reynolds). The MLP itself is not constrained; equivariance is the average. We do not canonicalize to one representative and run once.