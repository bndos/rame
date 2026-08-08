from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rame import EngineName, OrtSessionConfig, load_layout_model

from rame_benchmarks.models.model_meta import ModelMeta
from rame_benchmarks.models.models_protocols import LayoutPrediction
from rame_benchmarks.samples import ImageSample


class RameLayoutDetectionModel:
    def __init__(
        self,
        benchmark_model_meta: ModelMeta,
        *,
        source: str,
        engine: EngineName,
        engine_config: dict[str, Any] | None = None,
    ) -> None:
        self._benchmark_model_meta = benchmark_model_meta
        self._model = load_layout_model(
            source,
            model="pp-doclayout-plus",
            engine=engine,
            engine_config=engine_config_for(engine, engine_config),
        )

    @property
    def benchmark_model_meta(self) -> ModelMeta:
        return self._benchmark_model_meta

    def detect_layout_many(
        self, images: Sequence[ImageSample]
    ) -> Sequence[LayoutPrediction]:
        results = self._model.detect_layout_many(
            [sample.as_ndarray() for sample in images]
        )
        return [LayoutPrediction(regions=len(result.regions)) for result in results]


def engine_config_for(
    engine: EngineName,
    config: dict[str, Any] | None,
) -> OrtSessionConfig | None:
    if config is None:
        return None

    match engine:
        case "onnxruntime":
            return OrtSessionConfig(**config)
        case _:
            raise ValueError(f"unsupported engine {engine!r} for RAME models")
