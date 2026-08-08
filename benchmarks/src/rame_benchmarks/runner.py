from __future__ import annotations

from pathlib import Path
from typing import Any

from rame_benchmarks.models import ModelName, get_model
from rame_benchmarks.tasks import TaskName, TaskResult, get_tasks


def run_benchmark(
    *,
    model: ModelName,
    task_names: tuple[TaskName, ...] | None = None,
    output_folder: Path = Path("results"),
    batch_size: int = 32,
    warmup: int = 0,
    repeats: int = 1,
    overrides: dict[str, Any] | None = None,
) -> list[TaskResult]:
    tasks = get_tasks(task_names)
    data_folder = output_folder / "data"
    loaded_model = get_model(model, **(overrides or {}))
    results = []

    for task in tasks:
        task_output_folder = data_folder / task.name.value
        result = task.evaluate(
            loaded_model,
            task_output_folder,
            batch_size=batch_size,
            warmup=warmup,
            repeats=repeats,
        )
        results.append(result)

    return results
