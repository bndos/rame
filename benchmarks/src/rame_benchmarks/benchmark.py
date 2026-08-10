from __future__ import annotations

import json
import multiprocessing as mp
import traceback
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from rame_benchmarks.config import BenchmarkConfig
from rame_benchmarks.models import ModelName, get_model
from rame_benchmarks.tasks import TaskMetric, TaskName, TaskResult, get_tasks


@dataclass(frozen=True)
class Benchmark:
    model: ModelName
    task_names: tuple[TaskName, ...] | None = None
    output_folder: Path = Path("results")
    batch_size: int = 32
    warmup_batches: int = 0
    repeats: int = 1
    overrides: dict[str, Any] | None = None

    def run(self) -> BenchmarkResult:
        return BenchmarkResult(runs=[self.run_once()])

    def run_isolated_once(self, name: str) -> BenchmarkRunResult:
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_run_benchmark_child,
            args=(name, self, child_connection),
            name=f"rame-bench-{name}",
        )

        process.start()
        child_connection.close()
        process.join()

        if not parent_connection.poll():
            exitcode = process.exitcode
            process.close()
            raise RuntimeError(
                f"isolated benchmark run {name!r} exited without a result "
                f"(exit code {exitcode})"
            )

        message = parent_connection.recv()
        process.close()
        match message:
            case ("ok", result):
                return result
            case ("err", error_type, error_message, error_traceback):
                raise RuntimeError(
                    f"isolated benchmark run {name!r} failed with "
                    f"{error_type}: {error_message}\n{error_traceback}"
                )
            case _:
                raise RuntimeError(
                    f"isolated benchmark run {name!r} returned {message!r}"
                )

    def run_once(self, name: str | None = None) -> BenchmarkRunResult:
        tasks = get_tasks(self.task_names)
        data_folder = self.output_folder / "data"
        loaded_model = get_model(self.model, **(self.overrides or {}))
        results = []

        for task in tasks:
            task_output_folder = data_folder / task.name.value
            result = task.evaluate(
                loaded_model,
                task_output_folder,
                batch_size=self.batch_size,
                warmup_batches=self.warmup_batches,
                repeats=self.repeats,
            )
            results.append(result)

        return BenchmarkRunResult(name=name, model=self.model, results=results)


@dataclass(frozen=True)
class BenchmarkSuite:
    runs: dict[str, Benchmark]
    output: Path | None = None

    @classmethod
    def from_config(cls, config: BenchmarkConfig) -> BenchmarkSuite:
        return cls(
            runs={
                name: Benchmark(
                    model=run.model,
                    task_names=run.task_names,
                    output_folder=run.output_folder,
                    batch_size=run.batch_size,
                    warmup_batches=run.warmup_batches,
                    repeats=run.repeats,
                    overrides=run.overrides,
                )
                for name, run in config.resolved_runs().items()
            },
            output=config.output,
        )

    def run(self) -> BenchmarkResult:
        return BenchmarkResult(
            runs=[benchmark.run_once(name) for name, benchmark in self.runs.items()]
        )

    def run_isolated(self) -> BenchmarkResult:
        return BenchmarkResult(
            runs=[
                benchmark.run_isolated_once(name)
                for name, benchmark in self.runs.items()
            ]
        )


IsolatedRunMessage = tuple[str, Any] | tuple[str, str, str, str]


def _run_benchmark_child(
    name: str,
    benchmark: Benchmark,
    connection: Connection,
) -> None:
    try:
        result: IsolatedRunMessage = ("ok", benchmark.run_once(name))
    except Exception as err:
        result = (
            "err",
            type(err).__name__,
            str(err),
            traceback.format_exc(),
        )
    connection.send(result)
    connection.close()


@dataclass(frozen=True)
class BenchmarkRunResult:
    name: str | None
    model: ModelName
    results: list[TaskResult]

    def title(self) -> str:
        if self.name is None:
            return str(self.model)
        return f"{self.name} ({self.model})"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model": str(self.model),
            "results": [asdict(result) for result in self.results],
        }
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class BenchmarkResult:
    runs: list[BenchmarkRunResult]

    def print(self, console: Console | None = None) -> None:
        console = console or Console()
        for run in self.runs:
            for result in run.results:
                table = Table(title=f"{run.title()} on {result.task_name.value}")
                table.add_column("Metric")
                table.add_column("Value", justify="right")
                for metric in result.metrics:
                    table.add_row(metric.name, format_metric_value(metric))
                console.print(table)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        if len(self.runs) == 1 and self.runs[0].name is None:
            return self.runs[0].to_dict()
        return {"runs": [run.to_dict() for run in self.runs]}


def format_metric_value(metric: TaskMetric) -> str:
    value = (
        f"{metric.value:.3f}" if isinstance(metric.value, float) else str(metric.value)
    )
    if metric.unit is None:
        return value
    return f"{value} {metric.unit}"
