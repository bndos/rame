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
        device: str | None = None,
        model_dir: str | None = None,
        engine: str | None = None,
        engine_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        ensure_device_available(device=device, engine=engine)
        from paddlex import create_predictor  # noqa: PLC0415

        self._benchmark_model_meta = benchmark_model_meta
        predictor_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "device": device,
            "engine": engine,
            "engine_config": engine_config,
            **kwargs,
        }
        if model_dir is not None:
            predictor_kwargs["model_dir"] = model_dir

        self._predictor = create_predictor(**predictor_kwargs)

    @property
    def benchmark_model_meta(self) -> ModelMeta:
        return self._benchmark_model_meta

    def detect_layout_many(
        self, images: Sequence[ImageSample]
    ) -> Sequence[LayoutPrediction]:
        if not images:
            return []

        results = list(
            self._predictor.predict(
                [sample.as_ndarray() for sample in images],
                batch_size=len(images),
            )
        )
        return [LayoutPrediction(regions=len(result["boxes"])) for result in results]

    def close(self) -> None:
        self._predictor.close()


def ensure_device_available(*, device: str | None, engine: str | None) -> None:
    """Fail early when paddlex cannot use GPU.

    PaddleX can otherwise fail late in native code.
    """
    if device is None or not device.startswith("gpu"):
        return

    import paddle  # noqa: PLC0415

    if not paddle.device.is_compiled_with_cuda():
        raise RuntimeError(
            "Paddle GPU benchmark requested, but Paddle is not built with CUDA"
        )
    if paddle.device.cuda.device_count() == 0:
        raise RuntimeError(
            "Paddle GPU benchmark requested, but no CUDA device is available"
        )
    if engine != "hpi":
        return

    import ultra_infer  # noqa: PLC0415

    if (
        hasattr(ultra_infer, "is_built_with_gpu")
        and not ultra_infer.is_built_with_gpu()
    ):
        raise RuntimeError(
            "Paddle HPI GPU benchmark requested, but ultra_infer is not built with GPU support"
        )
