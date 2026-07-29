from __future__ import annotations

import typer

from rame_benchmarks.tasks import get_tasks

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def available_tasks() -> None:
    for task in get_tasks():
        typer.echo(task.name.value)
