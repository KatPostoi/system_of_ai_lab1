"""Configuration for the MNIST training pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
  """Hyperparameters and constants for MNIST training."""

  image_height: int = 28
  image_width: int = 28
  num_classes: int = 10
  hidden_units: int = 128
  batch_size: int = 32
  epochs: int = 10
  validation_split: float = 0.2
  random_seed: int = 42
  verbose: int = 1

  @property
  def input_dim(self) -> int:
    """Flattened number of input features for one image."""

    return self.image_height * self.image_width


DEFAULT_TRAINING_CONFIG = TrainingConfig()
