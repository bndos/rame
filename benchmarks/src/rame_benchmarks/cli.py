from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rame_benchmarks.models import MODEL_REGISTRY, ModelName, get_model_metas
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
) -> None:
    tasks = get_tasks(tuple(task_names) if task_names else None)
    data_folder = output_folder / "data"

    for task in tasks:
        task_output_folder = data_folder / task.name.value
        task.load_data(task_output_folder)
        typer.echo(
            f"{model} on {task.name.value}: loaded {len(task.images)} samples "
            f"(batch_size={batch_size})"
        )
