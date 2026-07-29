from __future__ import annotations

from rame_benchmarks.tasks.abstask import AbsTask, TaskName
from rame_benchmarks.tasks.layout import LayoutMicroTask, LayoutTask

TASKS: tuple[AbsTask, ...] = (LayoutTask(), LayoutMicroTask())


def get_tasks(names: tuple[TaskName, ...] | None = None) -> list[AbsTask]:
    tasks = list(TASKS)
    if names is None:
        return tasks
    selected = set(names)
    return [task for task in tasks if task.name in selected]
