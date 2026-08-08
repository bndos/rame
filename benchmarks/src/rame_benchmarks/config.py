from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from rame_benchmarks.models import ModelName
from rame_benchmarks.tasks import TaskName


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelName
    tasks: tuple[TaskName, ...] | None = None
    output_folder: Path = Path("results")
    batch_size: int = Field(default=32, gt=0)
    warmup: int = Field(default=0, ge=0)
    repeats: int = Field(default=1, gt=0)
    output: Path | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)

    @property
    def task_names(self) -> tuple[TaskName, ...] | None:
        return self.tasks


def load_run_config(path: Path) -> RunConfig:
    raw_config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw_config, Mapping):
        raise ValueError("benchmark config must be a mapping")

    return RunConfig.model_validate(cast(dict[str, Any], dict(raw_config)))
