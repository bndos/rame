from __future__ import annotations

from typing import Annotated

import typer

from rame_benchmarks.datasets import (
    DatasetName,
    supported_dataset,
)

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def dataset(
    name: Annotated[
        DatasetName,
        typer.Option("--dataset", help="Supported benchmark dataset."),
    ] = DatasetName.DOCLAYOUT_PUBLAYNET,
) -> None:
    dataset = supported_dataset(name)
    typer.echo(f"{dataset.name.value}: {dataset.repo_id} ({dataset.split})")
