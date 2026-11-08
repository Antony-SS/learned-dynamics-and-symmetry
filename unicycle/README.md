# Unicycle dynamics

Shared SE(2) geometry and symmetry wrappers. Robotarium experiments live in [`experiments/`](experiments/).

## Overview

`symmetry.py` wraps an arbitrary MLP. The network writes a body-frame increment (Δx, Δy, Δθ, Δv, Δω), sideslip included: the learned models get **no no-slip constraint**. Only the physics baseline fixes Δy = 0. We integrate on SE(2): (Δx, Δy) is lifted through the current heading, and the rate increments are added in the body frame. Commands are the Robotarium *commanded* (v, ω), not encoder-measured rates.

The pieces:

- `BodyMlp` (in `experiments/models.py`): a geometry-free MLP. Its output is (Δx, Δθ, Δv, Δω).
- `Quotient`: drops the arena pose. The features are (v_body, ω_body, v_cmd, ω_cmd).
- `SymmetricQuotientMLP`: wraps any 4 → 4 network and averages it over the left–right mirror (Z₂), which makes it exactly mirror-equivariant.
- `symmetrize_feature_stats`: mirror-invariant feature normalization. Required if you standardize the inputs of `SymmetricQuotientMLP` (see [Normalization](#normalization)).
- `apply_body_dx`: composes the body increment onto the current pose and rates:

```
x' = x + cosθ Δx − sinθ Δy
y' = y + sinθ Δx + cosθ Δy
θ' = wrap(θ + Δθ)
v' = v + Δv
ω' = ω + Δω
```

```python
from unicycle.symmetry import (
    SymmetricQuotientMLP, apply_body_dx, quotient_features, symmetrize_feature_stats,
)

model = SymmetricQuotientMLP(my_mlp)                   # any nn.Module, 4 → 5
mean, scale = symmetrize_feature_stats(mean, scale)    # from training features
dx = model((quotient_features(rates, command) - mean) / scale)
pose, rates = apply_body_dx(pose, rates, dx)
```

The analytic unicycle plant (the physics baseline) uses Δx = v_cmd dt, Δy = 0, Δθ = ω_cmd dt, and sets the next rates to the command. The learned families predict all five (Δx, Δy, Δθ, Δv, Δω).

## Math

The configuration lives in SE(2): position (x, y) and heading θ. The body twist is (v, ω). `Log` / `Exp` use the GTSAM Pose2 chart (ρx, ρy, φ). The training state is (x, y, θ, v_body, ω_body), with command (v_cmd, ω_cmd).

A body increment is written in the robot frame. The polar (Δx, Δy) rotates with the heading. Δθ, Δv and Δω stay in the body frame. The learned models are not told that Δy ≈ 0. (For reference, measured lateral body velocity in the Robotarium data is negligible: mean −0.5 mm/s, std 5 mm/s.)

## Continuous symmetries

Planar motion is invariant to translating and rotating the arena. Shifting (x, y), or adding a constant to θ and rotating the position with it, does not change the body-relative motion.

The quotient drops (x, y, θ). What remains are the body rates and the command, (v_body, ω_body, v_cmd, ω_cmd).

## Morphological symmetries

A differential-drive robot is unchanged by reflection through its body xz plane (left ↔ right).

**Polar** quantities (v, v_cmd, Δx, Δv) are unchanged, and sideslip Δy flips. **Axial** quantities (ω, ω_cmd, Δθ, Δω) flip sign (det = −1). Swapping the wheels, (left, right) → (right, left), is exactly "v stays, ω flips".

`SymmetricQuotientMLP` stacks the input with its mirror image, runs the MLP once on that doubled batch, maps each increment back with the matching polar and axial signs, and averages. The MLP itself is unconstrained. Equivariance comes from the average. We do not canonicalize to a single representative.

## Normalization

`SymmetricQuotientMLP` mirrors whatever it receives. If the features are standardized with ordinary per-channel statistics, the mirror is taken about the training mean of ω and ω_cmd instead of about zero, and the model is only approximately equivariant. With the robot 11 data the offset is about 0.01 rad/s, which matches the roughly 12° of heading drift over 30 s seen in the first trained `quotient_sym` checkpoint. `symmetrize_feature_stats` returns the statistics of the training data together with its mirror image: zero mean for ω and ω_cmd. `BodyStep.set_feature_stats` applies it automatically whenever the network is a `SymmetricQuotientMLP`.
