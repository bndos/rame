from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

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
def run(
    model: Annotated[
        str,
        typer.Option("-m", "--model", help="Model to run."),
    ],
    task_names: Annotated[
        list[TaskName] | None,
        typer.Option("--tasks", help="Tasks to run."),
    ] = None,
    output_folder: Annotated[
        Path,
        typer.Option("--output-folder", help="Output directory for benchmark results."),
    ] = Path("results"),
) -> None:
    tasks = get_tasks(tuple(task_names) if task_names else None)
    data_folder = output_folder / "data"

    for task in tasks:
        task_output_folder = data_folder / task.name.value
        task.load_data(task_output_folder)
        typer.echo(f"{model} on {task.name.value}: loaded {len(task.images)} samples")
