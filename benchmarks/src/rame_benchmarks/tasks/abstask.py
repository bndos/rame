from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from rame_benchmarks.models.models_protocols import BenchmarkModel


class TaskName(str, Enum):
    LAYOUT_THROUGHPUT = "layout-throughput"
    LAYOUT_THROUGHPUT_MICRO = "layout-throughput-micro"


@dataclass(frozen=True)
class DatasetMetadata:
    path: str
    split: str
    revision: str | None = None


@dataclass(frozen=True)
class TaskMetadata:
    name: TaskName
    dataset: DatasetMetadata


@dataclass(frozen=True)
class TaskMetric:
    name: str
    value: float | int
    unit: str | None = None


@dataclass(frozen=True)
class TaskResult:
    task_name: TaskName
    metrics: tuple[TaskMetric, ...]


class AbsTask(ABC):
    metadata: TaskMetadata

    def __init__(self) -> None:
        self.data_loaded = False

    @property
    def name(self) -> TaskName:
        return self.metadata.name

    @abstractmethod
    def load_data(self, output_dir: Path, **kwargs: Any) -> None: ...

    def evaluate(
        self,
        model: BenchmarkModel,
        output_dir: Path,
        *,
        batch_size: int,
        warmup_batches: int,
        repeats: int,
    ) -> TaskResult:
        if not self.data_loaded:
            self.load_data(output_dir)

        return self._evaluate(
            model,
            batch_size=batch_size,
            warmup_batches=warmup_batches,
            repeats=repeats,
        )

    @abstractmethod
    def _evaluate(
        self,
        model: BenchmarkModel,
        *,
        batch_size: int,
        warmup_batches: int,
        repeats: int,
    ) -> TaskResult: ...
