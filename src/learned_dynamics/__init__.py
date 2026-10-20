"""Shared supervised training for raw dynamics maps."""

from .model import DynamicsMLP
from .train import TrainResult, train_model

__all__ = ["DynamicsMLP", "TrainResult", "train_model"]
