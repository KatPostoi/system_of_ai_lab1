"""Data loading and preprocessing for MNIST."""

from __future__ import annotations

from typing import cast

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from src.ml_types import FeatureBatch, LabelBatch, OneHotBatch, PreparedMnistData, RawImageBatch, RawMnistData


def load_mnist_raw() -> RawMnistData:
  """Load MNIST as raw image and label arrays."""

  raw_data = mnist.load_data()
  ((x_train, y_train), (x_test, y_test)) = cast(
    tuple[tuple[RawImageBatch, LabelBatch], tuple[RawImageBatch, LabelBatch]],
    raw_data,
  )
  return RawMnistData(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


def preprocess_images(images: RawImageBatch, input_dim: int) -> FeatureBatch:
  """Flatten and normalize image pixels to ``float32`` range ``[0, 1]``."""

  processed = images.reshape(-1, input_dim).astype(np.float32) / 255.0
  return cast(FeatureBatch, processed)


def encode_labels(labels: LabelBatch, num_classes: int) -> OneHotBatch:
  """Convert integer labels to one-hot vectors with ``float32`` dtype."""

  encoded = to_categorical(labels, num_classes=num_classes).astype(np.float32)
  return cast(OneHotBatch, encoded)


def prepare_mnist_data(raw: RawMnistData, input_dim: int, num_classes: int) -> PreparedMnistData:
  """Prepare raw MNIST splits for training and evaluation."""

  x_train = preprocess_images(raw.x_train, input_dim=input_dim)
  x_test = preprocess_images(raw.x_test, input_dim=input_dim)
  y_train = encode_labels(raw.y_train, num_classes=num_classes)
  y_test = encode_labels(raw.y_test, num_classes=num_classes)
  return PreparedMnistData(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
