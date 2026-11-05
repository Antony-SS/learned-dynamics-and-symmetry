# Quadrotor Dynamics

Shared geometry and symmetry wrappers. The IDSIA Crazyflie benchmark lives in `[idsia/](idsia/)`.

## Overview

`symmetry.py` *wraps an arbitrary MLP. The net never sees SE(3) or the airframe group: it is a map **Rin → R12** that writes a **body** increment (Δp, Δv, Δθ, Δω).*

*Contract:*

- `NaiveResidual` *— geometry-free MLP. Output is always body (Δp, Δv, Δθ, Δω).*
- `Quotient` *— optional front end. World (R, v, ω, Ω) → (g_body, v_body, ω, Ω) (13-D). Drops p and yaw.*
- `Orbit` *— optional Reynolds average of a 13-D → 12-D map over the four xz / yz mirrors.*
- `apply_body_dx` *— reconstruct the next world state. Polar parts lift through the current R; rotation composes on SO(3):*

```
p' = p + R Δp
v' = v + R Δv
R' = R Exp(Δθ)
ω' = ω + Δω
```

`apply_body_dx` is the integrator: start from the current state (`SymmetryQuadDynamicsNet`, no vehicle model) or from frozen Crazyflie RK4 (`PhysQuotient` / `PhysQuotientSym`), then add that body increment.

## Math

### State & Increment

Rigid-body configuration is SE(3): position p ∈ R³ and attitude R ∈ SO(3). Rates are a twist: world velocity v and body angular velocity ω. SO(3) is handled with the Lie maps Exp: so(3) → SO(3) and Log: SO(3) → so(3) (Rodrigues). The 12-D chart used in training is (p, v, Log R, ω).

A body increment is a twist written in the body frame. Polar vectors (Δp, Δv) rotate with R; axial vectors (Δθ, Δω) stay in the body algebra. That is the only place the group acts on the leftover.

### Continuous symmetries

Free flight is invariant under horizontal translation and yaw: shifting p_xy or composing R ↦ R_z(ψ) R (and v with it) does not change the body-relative dynamics.

The quotient drops those coordinates. Gravity and velocity are pulled into the body,

```
g_body = Rᵀ e_z
v_body = Rᵀ v
```

and stacked with ω and the four rotor speeds Ω. The 13-D feature (g_body, v_body, ω, Ω) is unchanged if you translate the vehicle or yaw the world frame. Roll and pitch remain, through g_body.

### Morphological symmetries

A plus/X quadrotor is unchanged under reflection through the body xz and yz planes. Those two mirrors generate the group Z₂ × Z₂ (four elements).

For the resulting dynamics the corresponding output mapping is: **polar** (g, v, Δp, Δv) flip the reflected axis; **axial** (ω, Δθ, Δω) pick up the usual det = −1 signs. Rotor slots permute so the same motor stays in the same geometric corner (Crazyflie-X order in `ORBIT_MOTORS`).

`Orbit` evaluates the MLP on all four mirrored inputs, maps each 12-D output back with the matching polar/axial signs, and averages (Reynolds). If you permute Ω, the leftover (Δp, Δv, Δθ, Δω) permutes the same way. The MLP itself is not constrained; equivariance is the average.