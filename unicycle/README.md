# Unicycle leftovers

Shared SE(2) geometry and symmetry wrappers. Robotarium experiments live in `[experiments/](experiments/)`.

## Overview

`symmetry.py` wraps an arbitrary MLP. The net writes a **body** increment (Δx, Δy, Δθ). We integrate with the unicycle / SE(2) maps. Commands are always the Robotarium commanded `(v, ω)`, not encoder-realized rates.

Contract:

- `BodyMlp` — geometry-free MLP. Output is always the full body step (Δx, Δy, Δθ).
- `Quotient` — drop arena pose. Features are `(v_body, ω_body, v_cmd, ω_cmd)`.
- `Orbit` — Reynolds average over the left–right mirror (Z₂).
- `apply_body_dx` — compose the increment on SE(2):

```
x' = x + cosθ Δx − sinθ Δy
y' = y + sinθ Δx + cosθ Δy
θ' = wrap(θ + Δθ)
```

The analytic unicycle plant (physics baseline) is the same integrator with `Δx = v_cmd dt`, `Δy = 0`, `Δθ = ω_cmd dt`. Learned families predict the whole body step; they are not residuals on that plant.

## Math

Configuration is SE(2): position (x, y) and heading θ. The body twist is (v, ω). `Log` / `Exp` are the GTSAM Pose2 chart (ρx, ρy, φ). Training state is `(x, y, θ, v_body, ω_body)` with command `(v_cmd, ω_cmd)`.

A body increment is a twist written in the robot frame. Polar (Δx, Δy) rotate with heading; Δθ stays in the body algebra.

## Continuous symmetries

The plane is invariant under translation and rotation of the arena. Shifting (x, y) or adding a constant to θ (and rotating the position) does not change body-relative motion.

The quotient drops (x, y, θ). What remains is the body rates and the command, `(v_body, ω_body, v_cmd, ω_cmd)`.

## Morphological symmetries

A differential-drive robot is unchanged under reflection through the body xz plane (left–right).

**Polar** (v, Δx, Δy): v and Δx stay; Δy flips. **Axial / pseudo** (ω, Δθ, ω_cmd): pick up det = −1. Wheel slots permute so the same motor stays on the same side of the chassis (`ORBIT_WHEELS`: `(left, right)` → `(right, left)`). In `(v, ω)` that swap is exactly “v stays, ω flips”; we still go through the wheel map so the construction matches the quadrotor motor permute.

`Orbit` evaluates the MLP on both mirrored inputs, maps each 3-D leftover back with the matching polar / axial signs, and averages (Reynolds). The MLP itself is not constrained; equivariance is the average. We do not canonicalize to one representative and run once.