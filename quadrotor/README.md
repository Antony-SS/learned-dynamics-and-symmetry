# Quadrotor dynamics

Shared geometry and symmetry wrappers. The IDSIA Crazyflie benchmark lives in [`idsia/`](idsia/).

## Overview

`symmetry.py` wraps an arbitrary MLP. The network never sees SE(3) or the airframe group. It is a plain map ℝ¹³ → ℝ¹² that writes a **body-frame** increment (Δp, Δv, Δθ, Δω).

The pieces:

- `Quotient`: optional front end. Maps the world state (R, v, ω, Ω) to the 13-D quotient (g_body, v_body, ω, Ω). Drops position and yaw.
- `Orbit`: optional wrapper. Averages any 13-D → 12-D network over the four xz / yz mirrors, which makes it exactly mirror-equivariant.
- `symmetrize_feature_stats`: mirror-invariant feature normalization. Required if you standardize the inputs of `Orbit` (see [Normalization](#normalization)).
- `apply_body_dx`: rebuilds the next world state. Polar parts are lifted through the current attitude R, and the rotation composes on SO(3):

```
p' = p + R Δp
v' = v + R Δv
R' = R Exp(Δθ)
ω' = ω + Δω
```

The base step is either the current state (`SymmetryQuadDynamicsNet`, no vehicle model) or the frozen Crazyflie RK4 prediction (`PhysQuotient`, `PhysQuotientSym`). The body increment is added on top.

```python
from quadrotor.symmetry import Orbit, Quotient, apply_body_dx, symmetrize_feature_stats

model = Orbit(my_mlp)                                  # any nn.Module, 13 → 12
mean, scale = symmetrize_feature_stats(mean, scale)    # from training features
dx_body = model((Quotient()(x, u) - mean) / scale)
x_next = apply_body_dx(x_phys, x, dx_body)
```

## Math

### State and increment

The rigid-body configuration lives in SE(3): position p ∈ ℝ³ and attitude R ∈ SO(3). The rates are the world velocity v and the body angular velocity ω. SO(3) is handled with the Lie maps Exp: so(3) → SO(3) and Log: SO(3) → so(3) (Rodrigues). The 12-D training chart is (p, v, Log R, ω).

A body increment is written in the body frame. Polar vectors (Δp, Δv) rotate with R. Axial vectors (Δθ, Δω) stay in the body algebra. This lift is the only place the attitude acts on the increment.

### Continuous symmetries

Free flight is invariant to horizontal translation and yaw. Shifting p_xy, or composing R ↦ R_z(ψ) R (and rotating v with it), does not change the body-relative dynamics.

The quotient removes those coordinates. Gravity and velocity are expressed in the body frame,

```
g_body = Rᵀ e_z
v_body = Rᵀ v
```

and stacked with ω and the four rotor speeds Ω. The 13-D feature (g_body, v_body, ω, Ω) does not change when you translate the vehicle or yaw the world frame. Roll and pitch remain, through g_body.

### Morphological symmetries

An X-configuration quadrotor is unchanged by reflection through its body xz and yz planes. The two mirrors generate the Klein group Z₂ × Z₂ (four elements).

Under a mirror, **polar** vectors (g, v, Δp, Δv) flip the reflected axis, and **axial** vectors (ω, Δθ, Δω) pick up an extra det = −1. Rotor slots are permuted so that each rotor stays in the same geometric corner (Crazyflie-X order, `ORBIT_MOTORS`). A mirror also reverses each rotor's spin, so a mirrored CW rotor matches the CCW rotor it lands on.

`Orbit` evaluates the MLP on all four mirrored inputs, maps each output back with the matching polar and axial signs, and averages (the Reynolds operator). The MLP itself is unconstrained. Equivariance comes from the average.

### Normalization

`Orbit` mirrors whatever it receives. If the features are standardized with ordinary per-channel statistics, the mirror is taken about the training mean instead of about zero, and the four motors are compared on different scales. The model is then only approximately equivariant. `symmetrize_feature_stats` returns the statistics of the training data together with all four of its mirror images: zero mean on every sign-flipped channel, and one shared mean and std for the motors. `BodyLeftover.set_feature_stats` applies it automatically whenever the network is an `Orbit`.
