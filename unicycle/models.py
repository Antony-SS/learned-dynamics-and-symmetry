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

import torch

from unicycle.data import (
    COUNT_LIMIT,
    MAX_ANGULAR_VELOCITY,
    MAX_LINEAR_VELOCITY,
    ROBOT_DIAMETER,
    TIME_STEP,
    WINDOW,
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
