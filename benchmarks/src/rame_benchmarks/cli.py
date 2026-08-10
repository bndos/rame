from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rame_benchmarks.benchmark import Benchmark, BenchmarkResult, BenchmarkSuite
from rame_benchmarks.config import load_benchmark_config
from rame_benchmarks.models import MODEL_REGISTRY, ModelName, get_model_metas
from rame_benchmarks.overrides import parse_overrides
from rame_benchmarks.tasks import TaskName, get_tasks

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def available_tasks() -> None:
    for task in get_tasks():
        typer.echo(task.name.value)


@app.command()
def available_models() -> None:
    for model in get_model_metas():
        typer.echo(model.name)


def complete_model_names(incomplete: str) -> list[str]:
    return [name for name in MODEL_REGISTRY if name.startswith(incomplete)]


@app.command()
def run(
    model: Annotated[
        ModelName,
        typer.Option(
            "-m",
            "--model",
            help="Model to run.",
            autocompletion=complete_model_names,
        ),
    ],
    task_names: Annotated[
        list[TaskName] | None,
        typer.Option("--tasks", help="Tasks to run."),
    ] = None,
    output_folder: Annotated[
        Path,
        typer.Option("--output-folder", help="Output directory for benchmark results."),
    ] = Path("results"),
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size", help="Number of samples per model prediction batch."
        ),
    ] = 32,
    warmup_batches: Annotated[
        int,
        typer.Option("--warmup-batches", help="Untimed batches before measurement."),
    ] = 0,
    repeats: Annotated[
        int,
        typer.Option("--repeats", help="Timed full-dataset passes."),
    ] = 1,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON output path."),
    ] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Option(
            "-o",
            "--override",
            help="Model config override, e.g. engine_config.intra_op_num_threads=10.",
        ),
    ] = None,
) -> None:
    benchmark = Benchmark(
        model=model,
        task_names=tuple(task_names) if task_names else None,
        output_folder=output_folder,
        batch_size=batch_size,
        warmup_batches=warmup_batches,
        repeats=repeats,
        overrides=parse_overrides(overrides),
    )
    result = benchmark.run()
    result.print()
    if output is not None:
        result.write_json(output)


@app.command("run-config")
def run_config(
    config_path: Annotated[
        Path,
        typer.Argument(help="YAML benchmark config path."),
    ],
    run_name: Annotated[
        str | None,
        typer.Option("--run", help="Run only one named config row."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Override JSON output path."),
    ] = None,
) -> None:
    config = load_benchmark_config(config_path)
    suite = BenchmarkSuite.from_config(config)

    if run_name is not None:
        benchmark = suite.runs.get(run_name)
        if benchmark is None:
            raise typer.BadParameter(f"unknown benchmark run {run_name!r}")
        result = BenchmarkResult(runs=[benchmark.run_once(run_name)])
    elif config.isolate_runs:
        result = suite.run_isolated()
    else:
        result = suite.run()

    result.print()
    output_path = output if output is not None else suite.output
    if output_path is not None:
        result.write_json(output_path)
