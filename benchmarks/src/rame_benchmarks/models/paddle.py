from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rame_benchmarks.models.model_meta import ModelMeta
from rame_benchmarks.models.models_protocols import LayoutPrediction
from rame_benchmarks.samples import ImageSample


class PaddleLayoutDetectionModel:
    def __init__(
        self,
        benchmark_model_meta: ModelMeta,
        *,
        model_name: str,
        model_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        from paddlex import create_predictor  # noqa: PLC0415

        self._benchmark_model_meta = benchmark_model_meta
        predictor_kwargs: dict[str, Any] = {"model_name": model_name, **kwargs}
        if model_dir is not None:
            predictor_kwargs["model_dir"] = model_dir

        self._predictor = create_predictor(**predictor_kwargs)

    @property
    def benchmark_model_meta(self) -> ModelMeta:
        return self._benchmark_model_meta

    def detect_layout_many(
        self, images: Sequence[ImageSample], *, batch_size: int
    ) -> Sequence[LayoutPrediction]:
        if not images:
            return []

        results = list(
            self._predictor.predict(
                [sample.as_ndarray() for sample in images],
                batch_size=batch_size,
            )
        )
        return [LayoutPrediction(regions=len(result["boxes"])) for result in results]

    def close(self) -> None:
        self._predictor.close()
