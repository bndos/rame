from __future__ import annotations

from rame_benchmarks.tasks.abstask import AbsTask, TaskName
from rame_benchmarks.tasks.layout import LayoutTask
from rame_benchmarks.tasks.registry import TASKS, get_tasks

__all__ = [
    "AbsTask",
    "LayoutTask",
    "TASKS",
    "TaskName",
    "get_tasks",
]
