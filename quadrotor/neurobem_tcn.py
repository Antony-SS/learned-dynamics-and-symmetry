"""PyTorch reproduction of NeuroBEM's medium two-head TCN."""

from __future__ import annotations

import torch


class _TCNHead(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, 32, kernel_size=7),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv1d(32, 32, kernel_size=5),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 10, 20),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(20, 3),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.network(history.transpose(1, 2))


class NeuroBEMTCN(torch.nn.Module):
    """The paper's TCN-medium: 20 x 10 history and separate force/torque heads."""

    history_steps = 20
    input_dim = 10

    def __init__(self) -> None:
        super().__init__()
        self.force_head = _TCNHead(self.input_dim)
        self.torque_head = _TCNHead(self.input_dim)
        self.register_buffer("input_center", torch.zeros(self.input_dim))
        self.register_buffer("input_scale", torch.ones(self.input_dim))
        self.register_buffer("output_center", torch.zeros(6))
        self.register_buffer("output_scale", torch.ones(6))

    def set_normalization(
        self,
        input_center: torch.Tensor,
        input_scale: torch.Tensor,
        output_center: torch.Tensor,
        output_scale: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.input_center.copy_(input_center)
            self.input_scale.copy_(input_scale.clamp_min(1e-6))
            self.output_center.copy_(output_center)
            self.output_scale.copy_(output_scale.clamp_min(1e-6))

    def normalized_prediction(self, history: torch.Tensor) -> torch.Tensor:
        normalized = (history - self.input_center) / self.input_scale
        return torch.cat(
            (self.force_head(normalized), self.torque_head(normalized)),
            dim=-1,
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        normalized = self.normalized_prediction(history)
        return normalized * self.output_scale + self.output_center
