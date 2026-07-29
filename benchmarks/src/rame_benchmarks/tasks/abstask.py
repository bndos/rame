from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class TaskName(str, Enum):
    LAYOUT = "layout"


@dataclass(frozen=True)
class DatasetMetadata:
    path: str
    split: str
    revision: str | None = None


@dataclass(frozen=True)
class TaskMetadata:
    name: TaskName
    dataset: DatasetMetadata


class AbsTask(ABC):
    metadata: TaskMetadata

    def __init__(self) -> None:
        self.data_loaded = False
        self.images: list[Path] = []

    @property
    def name(self) -> TaskName:
        return self.metadata.name

    @abstractmethod
    def load_data(
        self,
        output_dir: Path,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> None: ...
