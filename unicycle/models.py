"""Feedforward maps from encoder increments to a body-frame step.

Every learned map remixes the scaled tick pair into ``(σ, δ)``, the
unicycle's natural coordinates, without the wheel radius or the track.
Naive also sees the arena pose. Symmetric maps then square and gate those
coordinates. Wheel radius, gear ratio, and track width stay in the wheel
baseline. They are not features.

The output is the tangent step ``(dx, dy, dθ)`` in meters and radians. Tick
inputs are divided by the full-scale count on one wheel. The last layer starts
at zero and is multiplied by one full-scale step, so the network begins at no
motion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from unicycle.data import (
    COUNT_LIMIT,
    DEFAULT_ROOT,
    MAX_ANGULAR_VELOCITY,
    MAX_LINEAR_VELOCITY,
    ROBOT_DIAMETER,
    TIME_STEP,
    TRAIN_LOGS,
    WINDOW,
    kept_steps,
    load_raw_named,
)

_LINEAR_STEP = MAX_LINEAR_VELOCITY * TIME_STEP * WINDOW
_ANGULAR_STEP = MAX_ANGULAR_VELOCITY * TIME_STEP * WINDOW


def _trunk(in_dim: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
    )


class _StepNet(torch.nn.Module):
    """Shared trunk, zero-initialized head, and fixed input and output scales."""

    uses_pose = False
    reads_counts = True

    def __init__(self, in_dim: int):
        super().__init__()
        self.trunk = _trunk(in_dim)
        self.head = torch.nn.Linear(16, 3)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.register_buffer("count_limit", torch.tensor(float(COUNT_LIMIT)))
        self.register_buffer("step_limit", torch.tensor([_LINEAR_STEP, _LINEAR_STEP, _ANGULAR_STEP]))

    def _scaled_head(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.head(encoded) * self.step_limit

    def _sigma_delta(self, features: torch.Tensor) -> torch.Tensor:
        scaled = features[..., :2] / self.count_limit
        total = scaled[..., 0] + scaled[..., 1]
        diff = scaled[..., 1] - scaled[..., 0]
        return torch.stack((total, diff), dim=-1)


class CanonicalModel(_StepNet):
    """Body step from the unicycle coordinates ``(σ, δ)``, ungated.

    Scale the tick pair, then ``σ = L̃ + R̃`` and ``δ = R̃ − L̃``. Those are
    ``v`` and ``ω`` up to the wheel radius and the track, which this map does
    not use. The three outputs are unconstrained. An SE(2) motion of the
    arena leaves the prediction unchanged. Rollout is
    ``T ← T ∘ Exp(forward)``.
    """

    def __init__(self):
        super().__init__(2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self._scaled_head(self.trunk(self._sigma_delta(features)))


class ResidualModel(CanonicalModel):
    """Same map as :class:`CanonicalModel`, including the ``(σ, δ)`` input.

    Training and rollout compose the wheel unicycle in front of this output,
    ``T ← T ∘ Z ∘ Exp(forward)``, with ``Z = Exp(v·dt, 0, ω·dt)`` built outside
    the network.     A zero output is exactly the wheel unicycle.
    """


class SymmetricModel(_StepNet):
    """Raw tick inputs, with a left-right mirror built into the layers.

    Swapping the wheels is the mirror. Write ``σ`` for the scaled sum of the
    two tick channels and ``δ`` for their scaled difference. The trunk sees
    ``(σ, δ²)``, and the lateral and yaw outputs are multiplied by ``δ``, so
    ``(Δleft, Δright) → (Δright, Δleft)`` sends
    ``(dx, dy, dθ) → (dx, −dy, −dθ)``. Rollout matches :class:`CanonicalModel`.
    """

    def __init__(self):
        super().__init__(2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scaled = features / self.count_limit
        total = scaled[..., 0] + scaled[..., 1]
        diff = scaled[..., 1] - scaled[..., 0]
        even = torch.stack((total, diff * diff), dim=-1)
        raw = self.head(self.trunk(even))
        odd = diff.unsqueeze(-1) * raw[..., 1:]
        return torch.cat((raw[..., :1], odd), dim=-1) * self.step_limit


def _sized_trunk(in_dim: int, hidden: int, encoded: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, encoded),
    )


class _MirrorBranch(torch.nn.Module):
    """One mirror-even or mirror-odd map on ``(σ, δ²)``.

    ``odd=False`` is the wheel-swap mirror: ``(dx, dy, dθ) → (dx, −dy, −dθ)``.
    ``odd=True`` is the other half: ``(dx, dy, dθ) → (−dx, dy, dθ)``. Their sum
    is an unconstrained step. Both heads start at zero.
    """

    def __init__(self, hidden: int, encoded: int, *, odd: bool):
        super().__init__()
        self.odd = bool(odd)
        self.trunk = _sized_trunk(2, hidden, encoded)
        self.head = torch.nn.Linear(encoded, 3)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.register_buffer("count_limit", torch.tensor(float(COUNT_LIMIT)))
        self.register_buffer("step_limit", torch.tensor([_LINEAR_STEP, _LINEAR_STEP, _ANGULAR_STEP]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scaled = features / self.count_limit
        total = scaled[..., 0] + scaled[..., 1]
        diff = scaled[..., 1] - scaled[..., 0]
        even = torch.stack((total, diff * diff), dim=-1)
        raw = self.head(self.trunk(even))
        signed = diff.unsqueeze(-1)
        if self.odd:
            mixed = torch.cat((signed * raw[..., :1], raw[..., 1:]), dim=-1)
        else:
            mixed = torch.cat((raw[..., :1], signed * raw[..., 1:]), dim=-1)
        return mixed * self.step_limit


class EvenOddModel(torch.nn.Module):
    """Canonical step written as a mirror-even network plus a mirror-odd network.

    Hidden width 22 and a 12-wide encoding give each branch about half the
    parameters of :class:`CanonicalModel`, so the sum matches that model.
    A zero head on both branches starts at the same place as a zero canonical
    head: no motion.

    ``asymmetry_penalty`` is the mean squared size of the odd branch, with yaw
    written as arc length at half the robot diameter, multiplied by
    ``penalty``. Training adds it to the position loss. The even branch is
    unpenalized.
    """

    uses_pose = False
    reads_counts = True

    def __init__(self, hidden: int = 22, encoded: int = 12, penalty: float = 2.0):
        super().__init__()
        self.even = _MirrorBranch(hidden, encoded, odd=False)
        self.odd = _MirrorBranch(hidden, encoded, odd=True)
        self.penalty = float(penalty)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.even(features) + self.odd(features)

    def asymmetry_penalty(self, features: torch.Tensor) -> torch.Tensor:
        odd = self.odd(features)
        arc = odd[..., 2] * (0.5 * ROBOT_DIAMETER)
        squared = odd[..., 0] ** 2 + odd[..., 1] ** 2 + arc ** 2
        return self.penalty * squared.mean()


class SymmetricResidualModel(SymmetricModel):
    """Left-right mirror on the correction that follows the wheel unicycle.

    The map matches :class:`SymmetricModel`: a wheel swap sends the correction
    to ``(dx, −dy, −dθ)``. Training composes it after the unicycle,
    ``T ← T ∘ Z ∘ Exp(forward)``. A zero output is exactly that unicycle.
    """


class ReversibleSymmetricModel(SymmetricModel):
    """Left-right mirror, and odd under reversing both wheels.

    Reversing both encoders sends ``(Δleft, Δright) → (−Δleft, −Δright)``.
    That is time reversal, ``(v, ω) → (−v, −ω)``, not a front-back geometric
    reflection. The step comes back negated,
    ``(dx, dy, dθ) → (−dx, −dy, −dθ)``, by averaging the symmetric map on the
    ticks with the same map on the negated ticks. Wheel-swap symmetry is
    inherited from :class:`SymmetricModel`. This forbids centrifugal
    ``v_y ~ vω`` slip and allows a rigid frame offset ``v_y ~ dω``.
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = super().forward(features)
        twin = super().forward(-features)
        return 0.5 * (raw - twin)


class FrontBackSymmetricModel(_StepNet):
    """Left-right wheel swap and a latitudinal (front-back) mirror through yz.

    The body twist is defined about the axle center. A sensor offset is a
    fixed frame transform, not part of this map. Write ``σ`` for the scaled
    sum of the tick channels and ``δ`` for their scaled difference. A
    left-right swap leaves ``σ`` even and ``δ`` odd. A front-back reflection
    negates both ticks, so both ``σ`` and ``δ`` flip. The trunk therefore
    sees ``(σ², δ²)``. It emits ``(a, b, c)``, and the step is

    ``Δx = σ a``, ``Δy = σδ b``, ``Δθ = δ c``.

    A wheel swap sends ``(dx, dy, dθ) → (dx, −dy, −dθ)``. Negating both
    ticks sends ``(dx, dy, dθ) → (−dx, +dy, −dθ)``. The ideal unicycle is
    constant ``a`` and ``c`` with ``b = 0``. Lateral slip vanishes when
    driving straight (``δ = 0``) and when spinning in place (``σ = 0``).
    """

    def __init__(self):
        super().__init__(2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scaled = features / self.count_limit
        total = scaled[..., 0] + scaled[..., 1]
        diff = scaled[..., 1] - scaled[..., 0]
        even = torch.stack((total * total, diff * diff), dim=-1)
        raw = self.head(self.trunk(even))
        dx = total.unsqueeze(-1) * raw[..., :1]
        dy = (total * diff).unsqueeze(-1) * raw[..., 1:2]
        dth = diff.unsqueeze(-1) * raw[..., 2:3]
        return torch.cat((dx, dy, dth), dim=-1) * self.step_limit


class ReversibleAsymModel(ReversibleSymmetricModel):
    """Forward-backward symmetric map plus a linear correction on the raw ticks.

    The extra head is zero at initialization and is not passed through the
    odd projection, so it can break the forward-backward constraint. Its size
    is penalized the same way as :class:`SymmetricAsymModel`.
    """

    def __init__(self, penalty: float = 1.0):
        super().__init__()
        self.asym_head = torch.nn.Linear(2, 3)
        torch.nn.init.zeros_(self.asym_head.weight)
        torch.nn.init.zeros_(self.asym_head.bias)
        self.penalty = float(penalty)

    def asymmetric(self, features: torch.Tensor) -> torch.Tensor:
        scaled = features / self.count_limit
        return self.asym_head(scaled) * self.step_limit

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return super().forward(features) + self.asymmetric(features)

    def asymmetry_penalty(self, features: torch.Tensor) -> torch.Tensor:
        asym = self.asymmetric(features)
        arc = asym[..., 2] * (0.5 * ROBOT_DIAMETER)
        squared = asym[..., 0] ** 2 + asym[..., 1] ** 2 + arc ** 2
        return self.penalty * squared.mean()


class SymmetricAsymModel(SymmetricModel):
    """Symmetric map plus a linear correction on the raw ticks.

    The extra head is zero at initialization, so the network starts as
    :class:`SymmetricModel`. ``asymmetry_penalty`` is the mean squared size of
    that correction, with yaw written as arc length at half the robot
    diameter, multiplied by ``penalty``. Training adds it to the position loss.
    """

    def __init__(self, penalty: float = 10.0):
        super().__init__()
        self.asym_head = torch.nn.Linear(2, 3)
        torch.nn.init.zeros_(self.asym_head.weight)
        torch.nn.init.zeros_(self.asym_head.bias)
        self.penalty = float(penalty)

    def asymmetric(self, features: torch.Tensor) -> torch.Tensor:
        scaled = features / self.count_limit
        return self.asym_head(scaled) * self.step_limit

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return super().forward(features) + self.asymmetric(features)

    def asymmetry_penalty(self, features: torch.Tensor) -> torch.Tensor:
        asym = self.asymmetric(features)
        arc = asym[..., 2] * (0.5 * ROBOT_DIAMETER)
        squared = asym[..., 0] ** 2 + asym[..., 1] ** 2 + arc ** 2
        return self.penalty * squared.mean()


_TAU_MIN = 1e-3
_TAU_MAX = 2.0


def fit_command_lag(root: str | Path | None = None) -> tuple[float, float]:
    """Least-squares first-order lags on native 30 Hz encoder ``ν`` and command ``u``.

    Uses the training prefix of each official training collection, before
    coarsening. Identifies the ODE ``ν̇ ≈ (u − ν) / τ`` at 30 Hz, separately
    for ``v`` and ``ω``. The plant then uses the exact sample of that ODE.
    """
    from unicycle.data import _train_val

    found = load_raw_named(DEFAULT_ROOT if root is None else root)
    logs = [_train_val(found[name])[0] for name in TRAIN_LOGS]
    xs_v: list[np.ndarray] = []
    ys_v: list[np.ndarray] = []
    xs_w: list[np.ndarray] = []
    ys_w: list[np.ndarray] = []
    for log in logs:
        velocity = np.asarray(log.increments, dtype=float) / log.dt
        command = np.asarray(log.commands, dtype=float)
        kept = kept_steps(log)
        ok = kept[:-1] & kept[1:]
        if not bool(ok.any()):
            continue
        accel = (velocity[1:] - velocity[:-1]) / log.dt
        error = command[:-1] - velocity[:-1]
        xs_v.append(error[ok, 0])
        ys_v.append(accel[ok, 0])
        xs_w.append(error[ok, 1])
        ys_w.append(accel[ok, 1])
    if not xs_v:
        raise ValueError("no kept consecutive steps to fit the command lag")
    return _lag_constant(np.concatenate(xs_v), np.concatenate(ys_v)), _lag_constant(
        np.concatenate(xs_w), np.concatenate(ys_w)
    )


_DELTA_EPS = 1e-6


def increment_scale(logs) -> tuple[float, float]:
    """Train-set std of encoder ``ν − u``, one scale per ``(v, ω)``.

    Only kept steps. Call this on the training split, never val or test.
    """
    rows: list[np.ndarray] = []
    for log in logs:
        velocity = np.asarray(log.increments, dtype=float) / log.dt
        command = np.asarray(log.commands, dtype=float)
        kept = kept_steps(log)
        if not bool(kept.any()):
            continue
        rows.append((velocity - command)[kept])
    if not rows:
        raise ValueError("no kept steps to fit the command residual scale")
    scale = np.concatenate(rows, axis=0).std(axis=0)
    scale = np.maximum(scale, _DELTA_EPS)
    return float(scale[0]), float(scale[1])


def _lag_constant(error: np.ndarray, accel: np.ndarray) -> float:
    gram = float(error @ error)
    if gram <= 0:
        raise ValueError("command lag fit has no command-velocity error")
    tau = gram / float(error @ accel)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError(f"command lag fit produced τ = {tau}")
    return float(np.clip(tau, _TAU_MIN, _TAU_MAX))


def history_slice(nu: np.ndarray, command: np.ndarray, start: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """``steps`` of ``(ν, u)`` ending at ``start``, left-padded if the log is short."""
    if steps < 1:
        raise ValueError(f"history steps must be positive, got {steps}")
    hist_start = max(0, int(start) - steps)
    nu_h = np.asarray(nu[hist_start:start], dtype=np.float32)
    command_h = np.asarray(command[hist_start:start], dtype=np.float32)
    if nu_h.shape[0] == 0:
        nu_h = np.asarray(nu[start : start + 1], dtype=np.float32)
        command_h = np.asarray(command[start : start + 1], dtype=np.float32)
    if nu_h.shape[0] < steps:
        pad = steps - nu_h.shape[0]
        nu_h = np.concatenate((np.repeat(nu_h[:1], pad, axis=0), nu_h), axis=0)
        command_h = np.concatenate((np.repeat(command_h[:1], pad, axis=0), command_h), axis=0)
    return nu_h, command_h


def rollout_command_velocity(
    model: torch.nn.Module,
    nu0: torch.Tensor,
    commands: torch.Tensor,
    dt: float,
    hidden: torch.Tensor | None = None,
    poses: torch.Tensor | None = None,
) -> torch.Tensor:
    """Open-loop body velocity. ``nu0`` is ``(B, 2)``, ``commands`` is ``(B, T, 2)``."""
    step = commands.new_tensor(dt)
    nu = nu0
    predicted = []
    recurrent = bool(getattr(model, "uses_hidden", False))
    uses_pose = bool(getattr(model, "uses_pose", False))
    if uses_pose and poses is None:
        raise ValueError("a pose-conditioned dynamics model needs a pose at each step")
    for time in range(commands.shape[1]):
        pose = None if poses is None else poses[:, time]
        if recurrent and uses_pose:
            nu, hidden = model.step_velocity(nu, commands[:, time], step, hidden, pose=pose)
        elif recurrent:
            nu, hidden = model.step_velocity(nu, commands[:, time], step, hidden)
        elif uses_pose:
            nu = model.step_velocity(nu, commands[:, time], step, pose=pose)
        else:
            nu = model.step_velocity(nu, commands[:, time], step)
        predicted.append(nu)
    return torch.stack(predicted, dim=1)


class CommandLagModel(torch.nn.Module):
    """First-order lag from commanded ``(v, ω)`` to actualized body velocity.

    ``ν_{k+1} = u_k + (ν_k − u_k) e^{−Δt/τ} + f_θ(ν_k, u_k)``. Pose is
    ``g_{k+1} = g_k ∘ Exp(Δt · ν_{k+1})``. ``f_θ`` is zero here. ``τ`` is
    fit once on 30 Hz logs and not trained.
    """

    uses_pose = False
    uses_velocity = True
    reads_counts = False
    reads_command = True

    def __init__(self, tau_v: float, tau_omega: float):
        super().__init__()
        if tau_v <= 0 or tau_omega <= 0:
            raise ValueError(f"lag constants must be positive, got {tau_v}, {tau_omega}")
        self.register_buffer("tau", torch.tensor([float(tau_v), float(tau_omega)]))

    def residual(self, nu: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        return nu.new_zeros(nu.shape)

    def step_velocity(self, nu: torch.Tensor, command: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        decay = torch.exp(-dt.reshape(-1, 1) / self.tau)
        return command + (nu - command) * decay + self.residual(nu, command)


class CommandCanonicalModel(CommandLagModel):
    """Lag plant plus an ungated residual ``f_θ(ν, u)``.

    The residual sees only body velocity and the command, so the velocity
    update is SE(2) invariant. The head starts at zero: epoch 0 is the lag.
    """

    def __init__(
        self,
        tau_v: float,
        tau_omega: float,
        delta_v: float | None = None,
        delta_omega: float | None = None,
    ):
        super().__init__(tau_v, tau_omega)
        self.trunk = _trunk(4)
        self.head = torch.nn.Linear(16, 2)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.register_buffer(
            "command_limit",
            torch.tensor([MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY]),
        )
        self.register_buffer(
            "residual_limit",
            torch.tensor(
                [
                    MAX_LINEAR_VELOCITY if delta_v is None else float(delta_v),
                    MAX_ANGULAR_VELOCITY if delta_omega is None else float(delta_omega),
                ]
            ),
        )

    def residual(self, nu: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        features = torch.cat((nu, command), dim=-1) / torch.cat((self.command_limit, self.command_limit))
        return self.head(self.trunk(features)) * self.residual_limit


class CommandSymmetricModel(CommandLagModel):
    """Lag plant plus a left–right symmetric residual on ``(ν, u)``.

    Even features are ``(v, v_c, ω², ω_c²)``. The yaw residual is
    ``a ω + b ω_c``, so flipping both rates flips only ``Δω``.
    """

    def __init__(
        self,
        tau_v: float,
        tau_omega: float,
        delta_v: float | None = None,
        delta_omega: float | None = None,
    ):
        super().__init__(tau_v, tau_omega)
        self.trunk = _trunk(4)
        self.head = torch.nn.Linear(16, 3)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.register_buffer(
            "command_limit",
            torch.tensor([MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY]),
        )
        self.register_buffer(
            "residual_limit",
            torch.tensor(
                [
                    MAX_LINEAR_VELOCITY if delta_v is None else float(delta_v),
                    MAX_ANGULAR_VELOCITY if delta_omega is None else float(delta_omega),
                ]
            ),
        )

    def residual(self, nu: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        scaled_nu = nu / self.command_limit
        scaled_u = command / self.command_limit
        speed, yaw = scaled_nu.unbind(dim=-1)
        commanded_speed, commanded_yaw = scaled_u.unbind(dim=-1)
        even = torch.stack((speed, commanded_speed, yaw * yaw, commanded_yaw * commanded_yaw), dim=-1)
        raw = self.head(self.trunk(even))
        delta_yaw = raw[..., 1] * yaw + raw[..., 2] * commanded_yaw
        return torch.stack((raw[..., 0], delta_yaw), dim=-1) * self.residual_limit


class DynamicsQuotient(torch.nn.Module):
    """Commanded leftover on the SE(2) quotient: ``ν_{k+1} = u_k + f_θ(ν_k, u_k)``.

    Features are body velocity and the command. Pose never enters, so the
    twist map is SE(2) invariant. The head starts at zero: epoch 0 is
    ``ν_{k+1} = u_k``. Pose integrates ``g ← g ∘ Exp(Δt ν_{k+1})``.
    """

    uses_pose = False
    uses_velocity = True
    reads_counts = False
    reads_command = True

    def __init__(self):
        super().__init__()
        self.trunk = _trunk(4)
        self.head = torch.nn.Linear(16, 2)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.register_buffer(
            "command_limit",
            torch.tensor([MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY]),
        )
        self.register_buffer(
            "residual_limit",
            torch.tensor([MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY]),
        )

    def residual(
        self,
        nu: torch.Tensor,
        command: torch.Tensor,
        pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = torch.cat((nu, command), dim=-1) / torch.cat((self.command_limit, self.command_limit))
        return self.head(self.trunk(features)) * self.residual_limit

    def step_velocity(
        self,
        nu: torch.Tensor,
        command: torch.Tensor,
        dt: torch.Tensor,
        pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return command + self.residual(nu, command, pose)


class DynamicsFullState(DynamicsQuotient):
    """Same leftover, but ``f_θ`` also sees arena pose ``(x, y, θ)``."""

    uses_pose = True

    def __init__(self):
        super().__init__()
        self.trunk = _trunk(7)
        self.register_buffer("pose_limit", torch.tensor([1.0, 1.0, torch.pi]))

    def residual(
        self,
        nu: torch.Tensor,
        command: torch.Tensor,
        pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pose is None:
            raise ValueError("full-state dynamics needs the pose at this step")
        yaw = (pose[..., 2] + torch.pi) % (2.0 * torch.pi) - torch.pi
        pose_f = torch.stack((pose[..., 0], pose[..., 1], yaw), dim=-1) / self.pose_limit
        twist_f = torch.cat((nu, command), dim=-1) / torch.cat((self.command_limit, self.command_limit))
        return self.head(self.trunk(torch.cat((pose_f, twist_f), dim=-1))) * self.residual_limit


class DynamicsQuotientSymmetry(DynamicsQuotient):
    """Quotient leftover with left–right (longitudinal-axis) parity.

    Even features are ``(v, v_c, ω², ω_c²)``. The yaw leftover is
    ``a ω + b ω_c``. Front–back is not enforced.
    """

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(16, 3)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def residual(
        self,
        nu: torch.Tensor,
        command: torch.Tensor,
        pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scaled_nu = nu / self.command_limit
        scaled_u = command / self.command_limit
        speed, yaw = scaled_nu.unbind(dim=-1)
        commanded_speed, commanded_yaw = scaled_u.unbind(dim=-1)
        even = torch.stack((speed, commanded_speed, yaw * yaw, commanded_yaw * commanded_yaw), dim=-1)
        raw = self.head(self.trunk(even))
        delta_yaw = raw[..., 1] * yaw + raw[..., 2] * commanded_yaw
        return torch.stack((raw[..., 0], delta_yaw), dim=-1) * self.residual_limit


class CommandCanonicalInstant(CommandCanonicalModel):
    """Command-anchored residual with no lag: ``ν_{k+1} = u_k + f_θ(ν_k, u_k)``."""

    def __init__(
        self,
        tau_v: float = 1.0,
        tau_omega: float = 1.0,
        delta_v: float | None = None,
        delta_omega: float | None = None,
    ):
        super().__init__(tau_v, tau_omega, delta_v=delta_v, delta_omega=delta_omega)

    def step_velocity(self, nu: torch.Tensor, command: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return command + self.residual(nu, command)


class CommandSymmetricGRU(CommandLagModel):
    """Lag plant plus a short left–right GRU residual.

    Hidden state is 0.2 s of even features ``(v, v_c, ω², ω_c²)``. The yaw
    leftover is ``a ω + b ω_c``, so a wheel swap flips only ``Δω``.
    """

    uses_hidden = True

    def __init__(
        self,
        tau_v: float,
        tau_omega: float,
        hidden: int = 8,
        history_s: float = 0.2,
        delta_v: float | None = None,
        delta_omega: float | None = None,
    ):
        super().__init__(tau_v, tau_omega)
        if hidden < 1:
            raise ValueError(f"hidden size must be positive, got {hidden}")
        if history_s <= 0:
            raise ValueError(f"history_s must be positive, got {history_s}")
        self.hidden_size = int(hidden)
        self.history_s = float(history_s)
        self.gru = torch.nn.GRUCell(4, self.hidden_size)
        self.head = torch.nn.Linear(self.hidden_size, 3)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.register_buffer(
            "command_limit",
            torch.tensor([MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY]),
        )
        self.register_buffer(
            "residual_limit",
            torch.tensor(
                [
                    MAX_LINEAR_VELOCITY if delta_v is None else float(delta_v),
                    MAX_ANGULAR_VELOCITY if delta_omega is None else float(delta_omega),
                ]
            ),
        )

    def history_steps(self, dt: float) -> int:
        return max(1, int(round(self.history_s / float(dt))))

    def zero_hidden(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)

    def _even_features(self, nu: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        scaled_nu = nu / self.command_limit
        scaled_u = command / self.command_limit
        speed, yaw = scaled_nu.unbind(dim=-1)
        commanded_speed, commanded_yaw = scaled_u.unbind(dim=-1)
        return torch.stack((speed, commanded_speed, yaw * yaw, commanded_yaw * commanded_yaw), dim=-1)

    def _residual_and_hidden(
        self,
        nu: torch.Tensor,
        command: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            hidden = self.zero_hidden(nu.shape[0], nu.device, nu.dtype)
        hidden = self.gru(self._even_features(nu, command), hidden)
        raw = self.head(hidden)
        yaw = nu[..., 1] / self.command_limit[1]
        commanded_yaw = command[..., 1] / self.command_limit[1]
        delta_yaw = raw[..., 1] * yaw + raw[..., 2] * commanded_yaw
        residual = torch.stack((raw[..., 0], delta_yaw), dim=-1) * self.residual_limit
        return residual, hidden

    def residual(self, nu: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        leftover, _hidden = self._residual_and_hidden(nu, command, None)
        return leftover

    def step_velocity(
        self,
        nu: torch.Tensor,
        command: torch.Tensor,
        dt: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decay = torch.exp(-dt.reshape(-1, 1) / self.tau)
        leftover, hidden = self._residual_and_hidden(nu, command, hidden)
        return command + (nu - command) * decay + leftover, hidden

    def warm_hidden(self, nu: torch.Tensor, command: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        """Teacher-forced GRU warmup. ``nu`` and ``command`` are ``(B, T, 2)``."""
        if hidden is None:
            hidden = self.zero_hidden(nu.shape[0], nu.device, nu.dtype)
        for time in range(nu.shape[1]):
            _leftover, hidden = self._residual_and_hidden(nu[:, time], command[:, time], hidden)
        return hidden


class CommandSymmetricInstantGRU(CommandSymmetricGRU):
    """Command-seeded symmetric GRU: ``ν_{k+1} = u_k + f_θ(ν_k, u_k, h_k)``."""

    def __init__(
        self,
        tau_v: float = 1.0,
        tau_omega: float = 1.0,
        hidden: int = 8,
        history_s: float = 0.2,
        delta_v: float | None = None,
        delta_omega: float | None = None,
    ):
        super().__init__(
            tau_v,
            tau_omega,
            hidden=hidden,
            history_s=history_s,
            delta_v=delta_v,
            delta_omega=delta_omega,
        )

    def step_velocity(
        self,
        nu: torch.Tensor,
        command: torch.Tensor,
        dt: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leftover, hidden = self._residual_and_hidden(nu, command, hidden)
        return command + leftover, hidden


class CommandSymmetricInstant(CommandSymmetricModel):
    """Symmetric command-anchored residual with no lag: ``ν_{k+1} = u_k + f_θ``."""

    def __init__(
        self,
        tau_v: float = 1.0,
        tau_omega: float = 1.0,
        delta_v: float | None = None,
        delta_omega: float | None = None,
    ):
        super().__init__(tau_v, tau_omega, delta_v=delta_v, delta_omega=delta_omega)

    def step_velocity(self, nu: torch.Tensor, command: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return command + self.residual(nu, command)


class CommandSymmetricInstantFrontBack(CommandSymmetricInstant):
    """Command-anchored residual with left–right and front–back parity.

    Left–right ``(v, ω, v_c, ω_c) → (v, −ω, v_c, −ω_c)`` sends
    ``(Δv, Δω) → (Δv, −Δω)``. Front–back (both wheels / both body axes)
    flips all four inputs and sends ``(Δv, Δω) → (−Δv, −Δω)``. Even
    features are ``(v², v_c², ω², ω_c²)``. Residuals are
    ``Δv = a v + b v_c`` and ``Δω = c ω + d ω_c``.
    """

    def __init__(
        self,
        tau_v: float = 1.0,
        tau_omega: float = 1.0,
        delta_v: float | None = None,
        delta_omega: float | None = None,
    ):
        super().__init__(tau_v, tau_omega, delta_v=delta_v, delta_omega=delta_omega)
        self.head = torch.nn.Linear(16, 4)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def residual(self, nu: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        scaled_nu = nu / self.command_limit
        scaled_u = command / self.command_limit
        speed, yaw = scaled_nu.unbind(dim=-1)
        commanded_speed, commanded_yaw = scaled_u.unbind(dim=-1)
        even = torch.stack(
            (
                speed * speed,
                commanded_speed * commanded_speed,
                yaw * yaw,
                commanded_yaw * commanded_yaw,
            ),
            dim=-1,
        )
        raw = self.head(self.trunk(even))
        delta_speed = raw[..., 0] * speed + raw[..., 1] * commanded_speed
        delta_yaw = raw[..., 2] * yaw + raw[..., 3] * commanded_yaw
        return torch.stack((delta_speed, delta_yaw), dim=-1) * self.residual_limit


class CommandSymmetricInstantNorm(CommandSymmetricInstant):
    """Command-anchored residual in train-set σ units: ``ν_{k+1} = u + σ_Δ ⊙ f_θ``."""

    def __init__(self, delta_v: float, delta_omega: float, tau_v: float = 1.0, tau_omega: float = 1.0):
        if delta_v <= 0 or delta_omega <= 0:
            raise ValueError(f"residual scales must be positive, got {delta_v}, {delta_omega}")
        super().__init__(tau_v, tau_omega, delta_v=delta_v, delta_omega=delta_omega)


class CommandSymmetricInstantSlip(torch.nn.Module):
    """Command-anchored residual with sideslip. ``ν = (v, v_y, ω)``.

    Anchor is ``(v_c, 0, ω_c)``. Left–right (wheel swap) flips ``v_y`` and
    ``ω``. Even features are ``(v, v_c, v_y², ω², ω_c²)``. Odd residuals are
    linear in ``(v_y, ω, ω_c)``.
    """

    uses_pose = False
    uses_velocity = True
    has_sideslip = True
    seeds_vicon = False
    reads_counts = False
    reads_command = True

    def __init__(self):
        super().__init__()
        self.trunk = _trunk(5)
        self.head = torch.nn.Linear(16, 7)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.register_buffer(
            "command_limit",
            torch.tensor([MAX_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY]),
        )
        self.register_buffer(
            "residual_limit",
            torch.tensor([MAX_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY]),
        )

    def residual(self, nu: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        scaled = nu / self.command_limit
        speed, slip, yaw = scaled.unbind(dim=-1)
        commanded_speed = command[..., 0] / self.command_limit[0]
        commanded_yaw = command[..., 1] / self.command_limit[2]
        even = torch.stack(
            (speed, commanded_speed, slip * slip, yaw * yaw, commanded_yaw * commanded_yaw),
            dim=-1,
        )
        raw = self.head(self.trunk(even))
        delta_speed = raw[..., 0]
        delta_slip = raw[..., 1] * slip + raw[..., 2] * yaw + raw[..., 3] * commanded_yaw
        delta_yaw = raw[..., 4] * slip + raw[..., 5] * yaw + raw[..., 6] * commanded_yaw
        return torch.stack((delta_speed, delta_slip, delta_yaw), dim=-1) * self.residual_limit

    def step_velocity(self, nu: torch.Tensor, command: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        zeros = command.new_zeros(command.shape[:-1])
        anchor = torch.stack((command[..., 0], zeros, command[..., 1]), dim=-1)
        return anchor + self.residual(nu, command)


class CommandSymmetricFree(torch.nn.Module):
    """Symmetric velocity map with no lag plant.

    Seeded from encoder ``(v, ω)``. Each step predicts the next body velocity
    from ``(ν, u)`` and integrates ``g ← g ∘ Exp(Δt · ν)``. Even features
    are ``(v, v_c, ω², ω_c²)``. Predicted yaw rate is ``a ω + b ω_c``.
    """

    uses_pose = False
    uses_velocity = True
    seeds_vicon = False
    reads_counts = False
    reads_command = True

    def __init__(self):
        super().__init__()
        self.trunk = _trunk(4)
        self.head = torch.nn.Linear(16, 3)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.register_buffer(
            "command_limit",
            torch.tensor([MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY]),
        )

    def step_velocity(self, nu: torch.Tensor, command: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        scaled_nu = nu / self.command_limit
        scaled_u = command / self.command_limit
        speed, yaw = scaled_nu.unbind(dim=-1)
        commanded_speed, commanded_yaw = scaled_u.unbind(dim=-1)
        even = torch.stack((speed, commanded_speed, yaw * yaw, commanded_yaw * commanded_yaw), dim=-1)
        raw = self.head(self.trunk(even))
        yaw_rate = raw[..., 1] * yaw + raw[..., 2] * commanded_yaw
        return torch.stack((raw[..., 0], yaw_rate), dim=-1) * self.command_limit


class NaiveModel(_StepNet):
    """Body step from ``(σ, δ)`` and the absolute pose ``(x, y, yaw)``.

    Yaw is wrapped to ``[-π, π)``. Pose stays in logged meters and radians.
    Moving the arena changes this input, so the map is not SE(2) invariant.
    The tick remix matches :class:`CanonicalModel`; the radius and the track
    still do not enter.
    """

    uses_pose = True

    def __init__(self):
        super().__init__(5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        yaw = (features[..., 4] + torch.pi) % (2.0 * torch.pi) - torch.pi
        pose = torch.stack((features[..., 2], features[..., 3], yaw), dim=-1)
        return self._scaled_head(self.trunk(torch.cat((self._sigma_delta(features), pose), dim=-1)))
