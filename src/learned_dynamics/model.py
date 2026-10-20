"""Default MLP for a raw one-step dynamics map."""

from __future__ import annotations

import torch


class DynamicsMLP(torch.nn.Module):
    """Map canonical features to the observed step, in physical units.

    The trunk matches the robotarium residual network, without a covariance
    head and without a zero-initialized residual. The output is the step itself.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
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
        # A single logged step is small. Start there so Adam is not undoing a
        # huge random guess. This is not a residual on a known model.
        torch.nn.init.zeros_(self.head.bias)
        torch.nn.init.uniform_(self.head.weight, -1e-3, 1e-3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(features))
