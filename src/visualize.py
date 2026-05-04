"""Visualization helpers for MNIST data and training metrics."""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from src.ml_types import LabelBatch, MetricHistory, RawImageBatch


def show_dataset_examples(images: RawImageBatch, labels: LabelBatch, num_classes: int, seed: int) -> None:
  """Display one random sample per class from the training split."""

  rng = random.Random(seed)
  figure, axes = plt.subplots(1, num_classes, figsize=(2.5 * num_classes, 3))
  axes_array = np.atleast_1d(axes)

  for class_index in range(num_classes):
    label_indexes = np.where(labels == class_index)[0]
    axis = axes_array[class_index]

    if label_indexes.size == 0:
      axis.set_title(f"Digit {class_index}\nno sample")
      axis.axis("off")
      continue

    sample_index = rng.choice(label_indexes.tolist())
    axis.imshow(images[sample_index], cmap="gray")
    axis.set_title(f"Цифра {class_index}")
    axis.axis("off")

  figure.tight_layout()
  plt.show()


def plot_training_history(history: MetricHistory) -> None:
  """Visualize model training metrics."""

  figure, axes = plt.subplots(1, 2, figsize=(12, 4))

  _plot_metric(axes[0], history, train_key="accuracy", validation_key="val_accuracy", title="Accuracy")
  _plot_metric(axes[1], history, train_key="loss", validation_key="val_loss", title="Loss")

  figure.tight_layout()
  plt.show()


def _plot_metric(
  axis: Axes,
  history: MetricHistory,
  *,
  train_key: str,
  validation_key: str,
  title: str,
) -> None:
  train_values = history.get(train_key, [])
  validation_values = history.get(validation_key, [])

  if train_values:
    axis.plot(train_values, label="Train")
  if validation_values:
    axis.plot(validation_values, label="Val")

  axis.set_title(title)
  if train_values or validation_values:
    axis.legend()
