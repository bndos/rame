from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rame_benchmarks.models.model_meta import ModelMeta
from rame_benchmarks.samples import ImageSample


@dataclass(frozen=True)
class LayoutPrediction:
    regions: int


class LayoutModelProtocol(Protocol):
    @property
    def benchmark_model_meta(self) -> ModelMeta: ...

    def detect_layout_many(
        self, images: Sequence[ImageSample], *, batch_size: int
    ) -> Sequence[LayoutPrediction]: ...


BenchmarkModel = LayoutModelProtocol
