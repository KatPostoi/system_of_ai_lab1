"""Domain types and contracts for the MNIST pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

RawImageBatch: TypeAlias = npt.NDArray[np.uint8]
LabelBatch: TypeAlias = npt.NDArray[np.uint8]
FeatureBatch: TypeAlias = npt.NDArray[np.float32]
OneHotBatch: TypeAlias = npt.NDArray[np.float32]
MetricHistory: TypeAlias = dict[str, list[float]]


@dataclass(frozen=True, slots=True)
class RawMnistData:
  """Raw train/test splits loaded from MNIST."""

  x_train: RawImageBatch
  y_train: LabelBatch
  x_test: RawImageBatch
  y_test: LabelBatch


@dataclass(frozen=True, slots=True)
class PreparedMnistData:
  """Preprocessed train/test splits ready for model training."""

  x_train: FeatureBatch
  y_train: OneHotBatch
  x_test: FeatureBatch
  y_test: OneHotBatch


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
  """Model evaluation metrics on the test split."""

  loss: float
  accuracy: float

@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
  """Training history and final evaluation metrics."""

  history: MetricHistory
  metrics: EvaluationMetrics


class HistoryLike(Protocol):
  """Minimal history contract used by plotting and reporting."""

  @property
  def history(self) -> dict[str, list[float]]: ...


class ModelLike(Protocol):
  """Minimal model interface required by the training pipeline."""

  def summary(self) -> None: ...

  def fit(
    self,
    x: FeatureBatch,
    y: OneHotBatch,
    *,
    batch_size: int,
    epochs: int,
    validation_split: float,
    verbose: int,
  ) -> HistoryLike: ...

  def evaluate(self, x: FeatureBatch, y: OneHotBatch, *, verbose: int) -> tuple[float, float]: ...


def to_metric_history(raw_history: dict[str, list[float]]) -> MetricHistory:
  """Convert Keras history values into plain ``float`` lists."""

  return {name: [float(value) for value in values] for name, values in raw_history.items()}


def to_float_pair(values: tuple[float, float]) -> tuple[float, float]:
  """Normalize numeric metric tuple values to builtin ``float``."""

  loss, accuracy = values
  return cast(float, float(loss)), cast(float, float(accuracy))
