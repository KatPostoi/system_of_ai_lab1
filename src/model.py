"""Model factory for MNIST classification."""

from __future__ import annotations

from typing import cast

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

from src.config import TrainingConfig
from src.ml_types import ModelLike


def build_model(config: TrainingConfig) -> ModelLike:
  """Build and compile a dense neural network for MNIST."""

  model = Sequential(
    [
      Input(shape=(config.input_dim,)),
      Dense(config.hidden_units, activation="relu"),
      Dense(config.num_classes, activation="softmax"),
    ]
  )
  model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
  return cast(ModelLike, model)
