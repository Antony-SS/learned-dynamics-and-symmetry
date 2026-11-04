"""Recurrent leftover nets (ParityMotorNet / GRU) for NeuroBEM experiments.

The network writes body-frame linear and angular acceleration. Position and
attitude are never predicted independently: the rollout integrates them from
velocity and angular velocity. ``mirror=True`` averages the complete recurrent
map over the left-right and front-back reflections of a Crazyflie X layout:

    forward +x, left +y

    M4 front-left     M1 front-right
    M3 back-left      M2 back-right

Polar vectors use the geometric reflection. Axial vectors (rotation and
angular velocity) also acquire the determinant sign. Motor positions are
permuted with the airframe. Averaging over this four-element group makes the
map exactly equivariant to both mirrors.
"""

from __future__ import annotations

import torch

# Broad but finite bounds for small-quadrotor accelerations.
_ACCELERATION_LIMIT = (40.0, 40.0, 40.0, 400.0, 400.0, 400.0)


def _mirror_features(features: torch.Tensor, name: str) -> torch.Tensor:
    gravity = features[..., 0:3]
    velocity = features[..., 3:6]
    omega = features[..., 6:9]
    motors = features[..., 9:13]
    invariant = features[..., 13:]
    if name == "lr":
        motors = motors[..., [3, 2, 1, 0]]
        gravity = gravity * gravity.new_tensor((1.0, -1.0, 1.0))
        velocity = velocity * velocity.new_tensor((1.0, -1.0, 1.0))
        omega = omega * omega.new_tensor((-1.0, 1.0, -1.0))
    elif name == "fb":
        motors = motors[..., [1, 0, 3, 2]]
        gravity = gravity * gravity.new_tensor((-1.0, 1.0, 1.0))
        velocity = velocity * velocity.new_tensor((-1.0, 1.0, 1.0))
        omega = omega * omega.new_tensor((1.0, -1.0, -1.0))
    elif name == "both":
        motors = motors[..., [2, 3, 0, 1]]
        gravity = gravity * gravity.new_tensor((-1.0, -1.0, 1.0))
        velocity = velocity * velocity.new_tensor((-1.0, -1.0, 1.0))
        omega = omega * omega.new_tensor((-1.0, -1.0, 1.0))
    else:
        raise ValueError(f"unknown mirror {name!r}")
    return torch.cat(
        (gravity, velocity, omega, motors, invariant),
        dim=-1,
    )


def _mirror_acceleration(acceleration: torch.Tensor, name: str) -> torch.Tensor:
    linear = acceleration[..., 0:3]
    angular = acceleration[..., 3:6]
    if name == "lr":
        polar = acceleration.new_tensor((1.0, -1.0, 1.0))
        axial = acceleration.new_tensor((-1.0, 1.0, -1.0))
    elif name == "fb":
        polar = acceleration.new_tensor((-1.0, 1.0, 1.0))
        axial = acceleration.new_tensor((1.0, -1.0, -1.0))
    elif name == "both":
        polar = acceleration.new_tensor((-1.0, -1.0, 1.0))
        axial = acceleration.new_tensor((-1.0, -1.0, 1.0))
    else:
        raise ValueError(f"unknown mirror {name!r}")
    return torch.cat((linear * polar, angular * axial), dim=-1)


class MotorNet(torch.nn.Module):
    """Map predicted state and motor history to body-frame accelerations."""

    reads_pose = False

    def __init__(
        self, in_dim: int, out_dim: int, mirror: bool = False, hidden_dim: int = 64
    ):
        super().__init__()
        if out_dim != 6:
            raise ValueError(f"expected 6 accelerations, got {out_dim}")
        if mirror and in_dim != 13:
            raise ValueError(f"the airframe mirror takes 13 canonical inputs, got {in_dim}")
        if in_dim not in (13, 16):
            raise ValueError(f"expected 13 or 16 inputs, got {in_dim}")
        self.mirror = bool(mirror)
        self.hidden_dim = int(hidden_dim)
        self.gru = torch.nn.GRUCell(in_dim, self.hidden_dim)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(64, out_dim)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.register_buffer("input_center", torch.zeros(in_dim, dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones(in_dim, dtype=torch.float32))
        self.register_buffer(
            "acceleration_limit",
            torch.tensor(_ACCELERATION_LIMIT, dtype=torch.float32),
        )

    def set_input_normalization(
        self, center: torch.Tensor, scale: torch.Tensor
    ) -> None:
        """Store training-split input statistics with the model checkpoint."""
        if center.shape != self.input_center.shape or scale.shape != self.input_scale.shape:
            raise ValueError(
                f"normalization must have shape {tuple(self.input_center.shape)}, "
                f"got {tuple(center.shape)} and {tuple(scale.shape)}"
            )
        with torch.no_grad():
            self.input_center.copy_(center.to(self.input_center))
            self.input_scale.copy_(scale.to(self.input_scale).clamp_min(1e-6))

    def initial_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        groups = 4 if self.mirror else 1
        return reference.new_zeros(batch_size, groups, self.hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.initial_hidden(features.shape[0], features)
        acceleration, _ = self.step(features, hidden)
        return acceleration

    def step(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        names = (None, "lr", "fb", "both") if self.mirror else (None,)
        outputs = []
        next_hidden = []
        for index, name in enumerate(names):
            branch_features = (
                features if name is None else _mirror_features(features, name)
            )
            branch_output, branch_hidden = self._raw_step(
                branch_features, hidden[:, index]
            )
            if name is not None:
                branch_output = _mirror_acceleration(branch_output, name)
            outputs.append(branch_output)
            next_hidden.append(branch_hidden)
        return torch.stack(outputs).mean(dim=0), torch.stack(next_hidden, dim=1)

    def _raw_step(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features - self.input_center) / self.input_scale
        hidden = self.gru(normalized, hidden)
        acceleration = torch.tanh(self.head(self.trunk(hidden)))
        return acceleration * self.acceleration_limit, hidden


class ParityMotorNet(torch.nn.Module):
    """One-pass recurrent map equivariant to both Crazyflie-X mirrors.

    Channels are typed by their signs under the left-right and front-back
    reflections: ``++``, ``-+``, ``+-``, and ``--``. Linear maps stay within
    a type, products combine types by XOR, and recurrent gates are invariant.
    This enforces the same Z2 x Z2 equivariance as orbit averaging without
    evaluating four transformed copies of the recurrent network.
    """

    reads_pose = False
    mirror = True

    def __init__(
        self, in_dim: int, out_dim: int, hidden_per_type: int = 32
    ) -> None:
        super().__init__()
        if in_dim != 13:
            raise ValueError(f"parity model expects 13 canonical inputs, got {in_dim}")
        if out_dim != 6:
            raise ValueError(f"expected 6 accelerations, got {out_dim}")
        self.hidden_per_type = int(hidden_per_type)
        if self.hidden_per_type < 1:
            raise ValueError("hidden_per_type must be positive")

        # Input dimensions after the motor Hadamard transform.
        type_dims = (3, 4, 4, 2)
        width = self.hidden_per_type
        self.input_maps = torch.nn.ModuleList(
            torch.nn.Linear(size, width, bias=(kind == 0))
            for kind, size in enumerate(type_dims)
        )
        self.state_maps = torch.nn.ModuleList(
            torch.nn.Linear(width, width, bias=(kind == 0))
            for kind in range(4)
        )

        # Unordered character products a*b=c. A product has exactly the XOR
        # parity of its two factors.
        self.product_pairs = (
            ((0, 0), (1, 1), (2, 2), (3, 3)),
            ((0, 1), (2, 3)),
            ((0, 2), (1, 3)),
            ((0, 3), (1, 2)),
        )
        self.product_maps = torch.nn.ModuleList(
            torch.nn.ModuleList(
                torch.nn.Linear(width, width, bias=(kind == 0))
                for _ in pairs
            )
            for kind, pairs in enumerate(self.product_pairs)
        )

        # Every gate is invariant, so it may safely scale hidden channels of
        # any parity. Squares retain directional magnitude without leaking sign.
        self.gates = torch.nn.ModuleList(
            torch.nn.Linear(4 * width, width) for _ in range(4)
        )
        self.output_maps = torch.nn.ModuleList(
            (
                torch.nn.Linear(width, 1, bias=True),   # az: ++
                torch.nn.Linear(width, 2, bias=False), # ay, alpha_x: -+
                torch.nn.Linear(width, 2, bias=False), # ax, alpha_y: +-
                torch.nn.Linear(width, 1, bias=False), # alpha_z: --
            )
        )
        for output in self.output_maps:
            torch.nn.init.zeros_(output.weight)
            if output.bias is not None:
                torch.nn.init.zeros_(output.bias)

        self.register_buffer("input_center", torch.zeros(in_dim, dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones(in_dim, dtype=torch.float32))
        self.register_buffer(
            "acceleration_limit",
            torch.tensor(_ACCELERATION_LIMIT, dtype=torch.float32),
        )

    def set_input_normalization(
        self, center: torch.Tensor, scale: torch.Tensor
    ) -> None:
        if center.shape != self.input_center.shape or scale.shape != self.input_scale.shape:
            raise ValueError(
                f"normalization must have shape {tuple(self.input_center.shape)}, "
                f"got {tuple(center.shape)} and {tuple(scale.shape)}"
            )
        with torch.no_grad():
            self.input_center.copy_(center.to(self.input_center))
            self.input_scale.copy_(scale.to(self.input_scale).clamp_min(1e-6))

    def initial_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(batch_size, 4, self.hidden_per_type)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.initial_hidden(features.shape[0], features)
        acceleration, _ = self.step(features, hidden)
        return acceleration

    @staticmethod
    def _typed_inputs(features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        gravity = features[..., 0:3]
        velocity = features[..., 3:6]
        omega = features[..., 6:9]
        motors = features[..., 9:13]

        # Orthonormal Hadamard coordinates for [FR, BR, BL, FL].
        collective = 0.5 * motors.sum(dim=-1, keepdim=True)
        left_right = 0.5 * (
            motors[..., 0:1] + motors[..., 1:2]
            - motors[..., 2:3] - motors[..., 3:4]
        )
        front_back = 0.5 * (
            motors[..., 0:1] - motors[..., 1:2]
            - motors[..., 2:3] + motors[..., 3:4]
        )
        diagonal = 0.5 * (
            motors[..., 0:1] - motors[..., 1:2]
            + motors[..., 2:3] - motors[..., 3:4]
        )
        return (
            torch.cat((gravity[..., 2:3], velocity[..., 2:3], collective), dim=-1),
            torch.cat((gravity[..., 1:2], velocity[..., 1:2], omega[..., 0:1], left_right), dim=-1),
            torch.cat((gravity[..., 0:1], velocity[..., 0:1], omega[..., 1:2], front_back), dim=-1),
            torch.cat((omega[..., 2:3], diagonal), dim=-1),
        )

    def step(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features - self.input_center) / self.input_scale
        typed = self._typed_inputs(normalized)
        sources = [
            self.input_maps[kind](typed[kind])
            + self.state_maps[kind](hidden[:, kind])
            for kind in range(4)
        ]
        candidates = []
        for kind, pairs in enumerate(self.product_pairs):
            value = sources[kind]
            for product_map, (left, right) in zip(self.product_maps[kind], pairs):
                value = value + product_map(sources[left] * sources[right])
            candidates.append(torch.tanh(value))

        invariant_gate_input = torch.cat(
            (sources[0], sources[1].square(), sources[2].square(), sources[3].square()),
            dim=-1,
        )
        next_hidden = []
        for kind in range(4):
            gate = torch.sigmoid(self.gates[kind](invariant_gate_input))
            next_hidden.append(
                (1.0 - gate) * hidden[:, kind] + gate * candidates[kind]
            )
        hidden_out = torch.stack(next_hidden, dim=1)

        even = self.output_maps[0](hidden_out[:, 0])
        lr_odd = self.output_maps[1](hidden_out[:, 1])
        fb_odd = self.output_maps[2](hidden_out[:, 2])
        both_odd = self.output_maps[3](hidden_out[:, 3])
        raw = torch.cat(
            (
                fb_odd[..., 0:1],
                lr_odd[..., 0:1],
                even,
                lr_odd[..., 1:2],
                fb_odd[..., 1:2],
                both_odd,
            ),
            dim=-1,
        )
        return torch.tanh(raw) * self.acceleration_limit, hidden_out


class ResidualParityMotorNet(ParityMotorNet):
    """Exact Z2 x Z2 map plus a tiny exactly antisymmetric residual.

    The residual is a memoryless MLP whose Reynolds average over the
    left-right / front-back group is subtracted, so the leftover cannot
    represent any equivariant map. The last layer starts at zero and a
    heavy regularizer is meant to keep this term small.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_per_type: int = 32,
        residual_width: int = 8,
        residual_scale: float = 0.05,
    ) -> None:
        super().__init__(in_dim, out_dim, hidden_per_type=hidden_per_type)
        self.residual_scale = float(residual_scale)
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(in_dim, residual_width),
            torch.nn.Tanh(),
            torch.nn.Linear(residual_width, out_dim),
        )
        torch.nn.init.zeros_(self.residual[-1].weight)
        torch.nn.init.zeros_(self.residual[-1].bias)
        self.last_residual: torch.Tensor | None = None
        self.last_symmetric: torch.Tensor | None = None
        self.residual_weight_l2 = 0.1
        self.residual_output_l2 = 5.0
        self._residual_energy: torch.Tensor | None = None
        self._residual_count = 0

    def reset_residual_stats(self) -> None:
        self._residual_energy = None
        self._residual_count = 0

    def residual_penalty(self, residual_energy: torch.Tensor, weight_l2: float, output_l2: float) -> torch.Tensor:
        weights = sum(parameter.square().sum() for parameter in self.residual.parameters())
        return weight_l2 * weights + output_l2 * residual_energy

    def _residual_logits(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.input_center) / self.input_scale
        return self.residual(normalized)

    def antisymmetric_residual(self, features: torch.Tensor) -> torch.Tensor:
        def raw_residual(sample: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self._residual_logits(sample)) * (
                self.acceleration_limit * self.residual_scale
            )

        raw = raw_residual(features)
        orbit = raw
        for name in ("lr", "fb", "both"):
            orbit = orbit + _mirror_acceleration(
                raw_residual(_mirror_features(features, name)),
                name,
            )
        return raw - orbit / 4

    def step(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acceleration, hidden = super().step(features, hidden)
        residual = self.antisymmetric_residual(features)
        self.last_symmetric = acceleration
        self.last_residual = residual
        energy = residual.square().mean()
        if self._residual_energy is None:
            self._residual_energy = energy
            self._residual_count = 1
        else:
            self._residual_energy = self._residual_energy + energy
            self._residual_count += 1
        return acceleration + residual, hidden


class _OrbitGRU(torch.nn.Module):
    """One compact GRU plus a linear acceleration head."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = torch.nn.GRUCell(in_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, 6)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def raw(
        self, normalized: torch.Tensor, hidden: torch.Tensor, limit: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.gru(normalized, hidden)
        return torch.tanh(self.head(hidden)) * limit, hidden


class EvenOddSplitNet(torch.nn.Module):
    """Two equal GRUs: one Reynolds-even, one Reynolds-odd.

    Each head is a compact GRU of the same width. The even head is the
    orbit average. The odd head is the same architecture with that
    average subtracted, so it cannot represent an equivariant map.
    """

    reads_pose = False
    mirror = True

    def __init__(self, in_dim: int, hidden_dim: int = 47) -> None:
        super().__init__()
        if in_dim != 13:
            raise ValueError(f"even/odd split expects 13 canonical inputs, got {in_dim}")
        self.hidden_dim = int(hidden_dim)
        self.even = _OrbitGRU(in_dim, self.hidden_dim)
        self.odd = _OrbitGRU(in_dim, self.hidden_dim)
        self.register_buffer("input_center", torch.zeros(in_dim, dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones(in_dim, dtype=torch.float32))
        self.register_buffer(
            "acceleration_limit",
            torch.tensor(_ACCELERATION_LIMIT, dtype=torch.float32),
        )
        self.last_symmetric: torch.Tensor | None = None
        self.last_residual: torch.Tensor | None = None

    def set_input_normalization(self, center: torch.Tensor, scale: torch.Tensor) -> None:
        if center.shape != self.input_center.shape or scale.shape != self.input_scale.shape:
            raise ValueError(
                f"normalization must have shape {tuple(self.input_center.shape)}, "
                f"got {tuple(center.shape)} and {tuple(scale.shape)}"
            )
        with torch.no_grad():
            self.input_center.copy_(center.to(self.input_center))
            self.input_scale.copy_(scale.to(self.input_scale).clamp_min(1e-6))

    def initial_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(batch_size, 2, 4, self.hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.initial_hidden(features.shape[0], features)
        acceleration, _ = self.step(features, hidden)
        return acceleration

    def _orbit(
        self,
        cell: _OrbitGRU,
        normalized: torch.Tensor,
        hidden: torch.Tensor,
        kind: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pulled = []
        next_hidden = []
        names = (None, "lr", "fb", "both")
        for index, name in enumerate(names):
            sample = normalized if name is None else _mirror_features(normalized, name)
            acceleration, state = cell.raw(sample, hidden[:, index], self.acceleration_limit)
            if name is not None:
                acceleration = _mirror_acceleration(acceleration, name)
            pulled.append(acceleration)
            next_hidden.append(state)
        stacked = torch.stack(pulled, dim=0)
        even = stacked.mean(dim=0)
        output = even if kind == "even" else stacked[0] - even
        return output, torch.stack(next_hidden, dim=1)

    def step(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features - self.input_center) / self.input_scale
        even, even_hidden = self._orbit(self.even, normalized, hidden[:, 0], "even")
        odd, odd_hidden = self._orbit(self.odd, normalized, hidden[:, 1], "odd")
        self.last_symmetric = even
        self.last_residual = odd
        return even + odd, torch.stack((even_hidden, odd_hidden), dim=1)


class _ParityPolynomialLayer(torch.nn.Module):
    """One equivariant nonlinear layer for the four Z2 x Z2 characters."""

    product_pairs = (
        ((0, 0), (1, 1), (2, 2), (3, 3)),
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = torch.nn.ModuleList(
            torch.nn.Linear(width, width, bias=(kind == 0))
            for kind in range(4)
        )
        self.products = torch.nn.ModuleList(
            torch.nn.ModuleList(
                torch.nn.Linear(width, width, bias=(kind == 0))
                for _ in pairs
            )
            for kind, pairs in enumerate(self.product_pairs)
        )

    def forward(
        self, channels: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        outputs = []
        for kind, pairs in enumerate(self.product_pairs):
            value = self.linear[kind](channels[kind])
            for product_map, (left, right) in zip(self.products[kind], pairs):
                value = value + product_map(channels[left] * channels[right])
            outputs.append(torch.tanh(value))
        return tuple(outputs)


class ParityMLP(torch.nn.Module):
    """Approximately 25k-parameter feedforward Z2 x Z2-equivariant dynamics."""

    reads_pose = False
    mirror = True

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 29,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if in_dim != 13:
            raise ValueError(f"parity MLP expects 13 canonical inputs, got {in_dim}")
        if out_dim != 6:
            raise ValueError(f"expected 6 accelerations, got {out_dim}")
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be positive")
        type_dims = (3, 4, 4, 2)
        self.input_maps = torch.nn.ModuleList(
            torch.nn.Linear(size, width, bias=(kind == 0))
            for kind, size in enumerate(type_dims)
        )
        self.layers = torch.nn.ModuleList(
            _ParityPolynomialLayer(width) for _ in range(depth)
        )
        self.output_maps = torch.nn.ModuleList(
            (
                torch.nn.Linear(width, 1, bias=True),
                torch.nn.Linear(width, 2, bias=False),
                torch.nn.Linear(width, 2, bias=False),
                torch.nn.Linear(width, 1, bias=False),
            )
        )
        for output in self.output_maps:
            torch.nn.init.zeros_(output.weight)
            if output.bias is not None:
                torch.nn.init.zeros_(output.bias)
        self.register_buffer("input_center", torch.zeros(in_dim, dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones(in_dim, dtype=torch.float32))
        self.register_buffer(
            "acceleration_limit",
            torch.tensor(_ACCELERATION_LIMIT, dtype=torch.float32),
        )

    def set_input_normalization(
        self, center: torch.Tensor, scale: torch.Tensor
    ) -> None:
        if center.shape != self.input_center.shape or scale.shape != self.input_scale.shape:
            raise ValueError(
                f"normalization must have shape {tuple(self.input_center.shape)}, "
                f"got {tuple(center.shape)} and {tuple(scale.shape)}"
            )
        with torch.no_grad():
            self.input_center.copy_(center.to(self.input_center))
            self.input_scale.copy_(scale.to(self.input_scale).clamp_min(1e-6))

    def initial_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(batch_size, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        acceleration, _ = self.step(
            features, self.initial_hidden(features.shape[0], features)
        )
        return acceleration

    def step(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features - self.input_center) / self.input_scale
        typed = ParityMotorNet._typed_inputs(normalized)
        channels: tuple[torch.Tensor, ...] = tuple(
            input_map(values)
            for input_map, values in zip(self.input_maps, typed)
        )
        for layer in self.layers:
            channels = layer(channels)
        even = self.output_maps[0](channels[0])
        lr_odd = self.output_maps[1](channels[1])
        fb_odd = self.output_maps[2](channels[2])
        both_odd = self.output_maps[3](channels[3])
        raw = torch.cat(
            (
                fb_odd[..., 0:1],
                lr_odd[..., 0:1],
                even,
                lr_odd[..., 1:2],
                fb_odd[..., 1:2],
                both_odd,
            ),
            dim=-1,
        )
        return torch.tanh(raw) * self.acceleration_limit, hidden


class ParityTemporalMLP(ParityMLP):
    """Equivariant MLP with a small explicit temporal state.

    ``fir`` augments the instantaneous input with a finite history of prior
    *rolled-out* state features and motor commands. ``ema`` replaces that
    shift register with learned exponential memories at several time scales.
    Both memories transform exactly like the input, so the parent model's
    Z2 x Z2 equivariance is preserved without orbit averaging or a GRU.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        memory: str = "ema",
        memory_size: int = 4,
        width: int = 24,
        depth: int = 2,
    ) -> None:
        if memory not in ("fir", "ema"):
            raise ValueError(f"memory must be 'fir' or 'ema', got {memory!r}")
        if memory_size < 1:
            raise ValueError("memory_size must be positive")
        super().__init__(in_dim, out_dim, width=width, depth=depth)
        self.memory = memory
        self.memory_size = int(memory_size)
        self.in_dim = int(in_dim)

        type_dims = (3, 4, 4, 2)
        copies = 1 + self.memory_size
        self.input_maps = torch.nn.ModuleList(
            torch.nn.Linear(size * copies, width, bias=(kind == 0))
            for kind, size in enumerate(type_dims)
        )
        if memory == "ema":
            # Initial update fractions span approximately 20--200 ms at
            # 100 Hz. They remain invariant learned scalars.
            update = torch.logspace(
                torch.log10(torch.tensor(0.4)),
                torch.log10(torch.tensor(0.05)),
                self.memory_size,
            )
            self.memory_logits = torch.nn.Parameter(
                torch.logit(update.clamp(1e-4, 1.0 - 1e-4))
            )
        else:
            self.register_parameter("memory_logits", None)

    def initial_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(batch_size, self.memory_size, self.in_dim)

    def step(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features - self.input_center) / self.input_scale
        if self.memory == "fir":
            memory = hidden
            next_hidden = torch.cat(
                (normalized.unsqueeze(1), hidden[:, :-1]),
                dim=1,
            )
        else:
            update = torch.sigmoid(self.memory_logits).view(1, -1, 1)
            memory = hidden + update * (normalized.unsqueeze(1) - hidden)
            next_hidden = memory

        typed_steps = [
            ParityMotorNet._typed_inputs(normalized),
            *[
                ParityMotorNet._typed_inputs(memory[:, index])
                for index in range(self.memory_size)
            ],
        ]
        typed = tuple(
            torch.cat([step[kind] for step in typed_steps], dim=-1)
            for kind in range(4)
        )
        channels: tuple[torch.Tensor, ...] = tuple(
            input_map(values)
            for input_map, values in zip(self.input_maps, typed)
        )
        for layer in self.layers:
            channels = layer(channels)

        even = self.output_maps[0](channels[0])
        lr_odd = self.output_maps[1](channels[1])
        fb_odd = self.output_maps[2](channels[2])
        both_odd = self.output_maps[3](channels[3])
        raw = torch.cat(
            (
                fb_odd[..., 0:1],
                lr_odd[..., 0:1],
                even,
                lr_odd[..., 1:2],
                fb_odd[..., 1:2],
                both_odd,
            ),
            dim=-1,
        )
        return torch.tanh(raw) * self.acceleration_limit, next_hidden
