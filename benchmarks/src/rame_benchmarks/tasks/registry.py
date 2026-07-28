from __future__ import annotations

from enum import Enum

from rame_benchmarks.datasets import DOCLAYOUT_PUBLAYNET, BenchmarkDataset


class TaskName(str, Enum):
    LAYOUT = "layout"


class BenchmarkTask:
    name: TaskName
    datasets: tuple[BenchmarkDataset, ...]


class LayoutTask(BenchmarkTask):
    name = TaskName.LAYOUT
    datasets = (DOCLAYOUT_PUBLAYNET,)


TASKS: tuple[BenchmarkTask, ...] = (LayoutTask(),)


def get_tasks(names: tuple[TaskName, ...] | None = None) -> list[BenchmarkTask]:
    tasks = list(TASKS)
    if names is None:
        return tasks

    selected = set(names)
    return [task for task in tasks if task.name in selected]
