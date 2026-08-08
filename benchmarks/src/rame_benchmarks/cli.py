from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rame_benchmarks.config import load_run_config
from rame_benchmarks.models import MODEL_REGISTRY, ModelName, get_model_metas
from rame_benchmarks.overrides import parse_overrides
from rame_benchmarks.reporting import print_task_results, write_task_results_json
from rame_benchmarks.runner import run_benchmark
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
    warmup: Annotated[
        int,
        typer.Option("--warmup", help="Untimed full-dataset warmup passes."),
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
    results = run_benchmark(
        model=model,
        task_names=tuple(task_names) if task_names else None,
        output_folder=output_folder,
        batch_size=batch_size,
        warmup=warmup,
        repeats=repeats,
        overrides=parse_overrides(overrides),
    )
    print_task_results(model, results)
    if output is not None:
        write_task_results_json(output, model, results)


@app.command("run-config")
def run_config(
    config_path: Annotated[
        Path,
        typer.Argument(help="YAML benchmark config path."),
    ],
) -> None:
    config = load_run_config(config_path)
    results = run_benchmark(
        model=config.model,
        task_names=config.task_names,
        output_folder=config.output_folder,
        batch_size=config.batch_size,
        warmup=config.warmup,
        repeats=config.repeats,
        overrides=config.overrides,
    )
    print_task_results(config.model, results)
    if config.output is not None:
        write_task_results_json(config.output, config.model, results)
