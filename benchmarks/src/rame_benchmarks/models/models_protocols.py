from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rame_benchmarks.models.registry import ModelMeta


@dataclass(frozen=True)
class LayoutPrediction:
    regions: int


class LayoutModelProtocol(Protocol):
    @property
    def benchmark_model_meta(self) -> ModelMeta: ...

    def detect_layout_many(
        self, images: Sequence[Path]
    ) -> Sequence[LayoutPrediction]: ...


BenchmarkModel = LayoutModelProtocol
