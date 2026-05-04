"""Application entrypoint for the MNIST lab."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path


def _resolve_entrypoint() -> Callable[[], object]:
  """Load the training entrypoint for both script and module execution."""

  if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
      sys.path.insert(0, project_root_str)

  from src.task_1 import train_and_evaluate

  return train_and_evaluate


if __name__ == "__main__":
  print("Запуск лабораторной работы: Распознавание MNIST")
  _resolve_entrypoint()()
