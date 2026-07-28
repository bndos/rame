from __future__ import annotations

import typer

from rame_benchmarks.tasks import get_tasks

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def tasks() -> None:
    for task in get_tasks():
        datasets = ", ".join(dataset.name.value for dataset in task.datasets)
        typer.echo(f"{task.name.value}: datasets={datasets}")
