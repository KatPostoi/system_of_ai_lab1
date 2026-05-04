"""Facade module kept for lab entrypoint compatibility."""

from __future__ import annotations

from src.config import DEFAULT_TRAINING_CONFIG
from src.ml_types import TrainingArtifacts
from src.train import run_training_pipeline


def train_and_evaluate() -> TrainingArtifacts:
  """Run the full MNIST lab workflow and return training artifacts."""

  return run_training_pipeline(DEFAULT_TRAINING_CONFIG)
