from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from rame_benchmarks.models import ModelName
from rame_benchmarks.tasks import TaskName


class RunDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: tuple[TaskName, ...] | None = None
    output_folder: Path = Path("results")
    batch_size: int = Field(default=32, gt=0)
    warmup_batches: int = Field(default=0, ge=0)
    repeats: int = Field(default=1, gt=0)

    @property
    def task_names(self) -> tuple[TaskName, ...] | None:
        return self.tasks


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelName
    tasks: tuple[TaskName, ...] | None = None
    output_folder: Path | None = None
    batch_size: int | None = Field(default=None, gt=0)
    warmup_batches: int | None = Field(default=None, ge=0)
    repeats: int | None = Field(default=None, gt=0)
    overrides: dict[str, Any] = Field(default_factory=dict)

    def resolve(self, defaults: RunDefaults) -> ResolvedRunConfig:
        return ResolvedRunConfig(
            model=self.model,
            tasks=self.tasks if self.tasks is not None else defaults.tasks,
            output_folder=self.output_folder or defaults.output_folder,
            batch_size=self.batch_size or defaults.batch_size,
            warmup_batches=defaults.warmup_batches
            if self.warmup_batches is None
            else self.warmup_batches,
            repeats=self.repeats or defaults.repeats,
            overrides=self.overrides,
        )


class ResolvedRunConfig(RunDefaults):
    model: ModelName
    overrides: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: RunDefaults = Field(default_factory=RunDefaults)
    output: Path | None = None
    runs: dict[str, RunConfig]

    def resolved_runs(self) -> dict[str, ResolvedRunConfig]:
        return {name: run.resolve(self.defaults) for name, run in self.runs.items()}


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    raw_config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw_config, Mapping):
        raise ValueError("benchmark config must be a mapping")

    return BenchmarkConfig.model_validate(cast(dict[str, Any], dict(raw_config)))
