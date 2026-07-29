from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path


class TaskName(str, Enum):
    LAYOUT = "layout"


class AbsTask(ABC):
    name: TaskName
    data_loaded: bool = False

    @abstractmethod
    def load_data(
        self,
        output_dir: Path,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> None: ...
