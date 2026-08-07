from __future__ import annotations

from collections.abc import Sequence

from rame import load_layout_model

from rame_benchmarks.models.model_meta import ModelMeta
from rame_benchmarks.models.models_protocols import LayoutPrediction
from rame_benchmarks.samples import ImageSample


class RameLayoutDetectionModel:
    def __init__(self, benchmark_model_meta: ModelMeta) -> None:
        self._benchmark_model_meta = benchmark_model_meta
        self._model = load_layout_model(
            "PaddlePaddle/PP-DocLayout_plus-L_onnx",
            model="pp-doclayout-plus",
            engine="onnx",
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
