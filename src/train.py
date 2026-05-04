"""Training orchestration for MNIST."""

from __future__ import annotations

import random

import numpy as np
import tensorflow as tf

from src.config import DEFAULT_TRAINING_CONFIG, TrainingConfig
from src.data import load_mnist_raw, prepare_mnist_data
from src.ml_types import EvaluationMetrics, TrainingArtifacts, to_float_pair, to_metric_history
from src.model import build_model
from src.visualize import plot_training_history, show_dataset_examples


def set_global_seed(seed: int) -> None:
  """Set all random seeds used by the pipeline."""

  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)


def run_training_pipeline(config: TrainingConfig = DEFAULT_TRAINING_CONFIG) -> TrainingArtifacts:
  """Run the full MNIST workflow: visualize, train, evaluate, and report."""

  set_global_seed(config.random_seed)

  raw_data = load_mnist_raw()
  show_dataset_examples(
    images=raw_data.x_train,
    labels=raw_data.y_train,
    num_classes=config.num_classes,
    seed=config.random_seed,
  )

  prepared_data = prepare_mnist_data(raw=raw_data, input_dim=config.input_dim, num_classes=config.num_classes)
  model = build_model(config)
  model.summary()

  history_object = model.fit(
    prepared_data.x_train,
    prepared_data.y_train,
    batch_size=config.batch_size,
    epochs=config.epochs,
    validation_split=config.validation_split,
    verbose=config.verbose,
  )

  metric_history = to_metric_history(history_object.history)
  test_loss, test_accuracy = to_float_pair(model.evaluate(prepared_data.x_test, prepared_data.y_test, verbose=0))

  metrics = EvaluationMetrics(loss=test_loss, accuracy=test_accuracy)
  print(f"Точность: {metrics.accuracy:.4f}")
  plot_training_history(metric_history)

  return TrainingArtifacts(history=metric_history, metrics=metrics)
