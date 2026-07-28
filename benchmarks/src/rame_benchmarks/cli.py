from __future__ import annotations

from typing import Annotated

import typer

from rame_benchmarks.tasks import TaskName, get_tasks

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def task(
    name: Annotated[
        TaskName,
        typer.Option("--task", help="Supported benchmark task."),
    ] = TaskName.LAYOUT,
) -> None:
    [task] = get_tasks((name,))
    datasets = ", ".join(dataset.name.value for dataset in task.datasets)
    typer.echo(f"{task.name.value}: datasets={datasets}")
