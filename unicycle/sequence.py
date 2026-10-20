"""Wheel-command networks for the unicycle step.

A feedforward map of the current ``(v·dt, ω·dt)``.
"""

from __future__ import annotations

import torch

from unicycle.data import MAX_ANGULAR_VELOCITY, MAX_LINEAR_VELOCITY, TIME_STEP, WINDOW


class CommandNet(torch.nn.Module):
    """Map wheel steps to a correction in meters and radians.

    Commands enter as ``v / v_max`` and ``ω / ω_max``, using the Robotarium
    actuator caps. The head starts at zero and is multiplied by one full-scale
    window, so a zero head adds nothing. Each step is scored on its own.
    """

    def __init__(self, in_dim: int, out_dim: int, mirror: bool = False):
        super().__init__()
        if in_dim != 2 or out_dim != 3:
            raise ValueError(f"expected wheel input 2 and tangent output 3, got {in_dim}, {out_dim}")
        self.mirror = bool(mirror)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
        )
        self.head = torch.nn.Linear(16, out_dim)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        linear_step = MAX_LINEAR_VELOCITY * TIME_STEP * WINDOW
        angular_step = MAX_ANGULAR_VELOCITY * TIME_STEP * WINDOW
        self.register_buffer(
            "command_limit",
            torch.tensor([linear_step, angular_step], dtype=torch.float32),
        )
        self.register_buffer(
            "step_limit",
            torch.tensor([linear_step, linear_step, angular_step], dtype=torch.float32),
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Step in meters and radians. ``features`` is ``(F,)``, ``(T, F)``, or ``(B, T, F)``.

        With ``mirror=True`` the trunk sees ``(v, ω²)`` and the lateral and yaw
        outputs are multiplied by ``ω``. That is the left-right reflection:
        ``(v, ω) → (v, -ω)`` sends ``(x, y, φ) → (x, -y, -φ)``.
        """
        if self.mirror:
            return self._mirror(features)
        return self._raw(features)

    def _mirror(self, features: torch.Tensor) -> torch.Tensor:
        scaled = features / self.command_limit
        signed_yaw = scaled[..., 1]
        even = torch.stack((scaled[..., 0], signed_yaw * signed_yaw), dim=-1)
        raw = self.head(self.trunk(even))
        odd = signed_yaw.unsqueeze(-1) * raw[..., 1:]
        return torch.cat((raw[..., :1], odd), dim=-1) * self.step_limit

    def _raw(self, features: torch.Tensor) -> torch.Tensor:
        scaled = features / self.command_limit
        return self.head(self.trunk(scaled)) * self.step_limit
