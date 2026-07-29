from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rame_benchmarks.models.model_meta import ModelMeta


@dataclass(frozen=True)
class LayoutPrediction:
    regions: int


class LayoutModelProtocol(Protocol):
    @property
    def benchmark_model_meta(self) -> ModelMeta[LayoutModelProtocol]: ...

    def detect_layout_many(
        self, images: Sequence[Path]
    ) -> Sequence[LayoutPrediction]: ...


BenchmarkModel = LayoutModelProtocol
