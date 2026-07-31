from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rame_benchmarks.models import ModelName
from rame_benchmarks.tasks import TaskMetric, TaskResult


def print_task_results(
    model: ModelName,
    results: Sequence[TaskResult],
    console: Console | None = None,
) -> None:
    console = console or Console()
    for result in results:
        table = Table(title=f"{model} on {result.task_name.value}")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for metric in result.metrics:
            table.add_row(metric.name, format_metric_value(metric))
        console.print(table)


def format_metric_value(metric: TaskMetric) -> str:
    value = (
        f"{metric.value:.3f}" if isinstance(metric.value, float) else str(metric.value)
    )
    if metric.unit is None:
        return value
    return f"{value} {metric.unit}"


def write_task_results_json(
    path: Path,
    model: ModelName,
    results: Sequence[TaskResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(model),
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
