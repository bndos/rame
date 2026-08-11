from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rame import (
    CpuExecutionProviderConfig,
    CudaExecutionProviderConfig,
    EngineName,
    OrtSessionConfig,
    TensorRtExecutionProviderConfig,
    load_layout_model,
)

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
        copy: bool = True,
    ) -> None:
        self._benchmark_model_meta = benchmark_model_meta
        self._copy = copy
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
            [sample.as_ndarray() for sample in images], copy=self._copy
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
            return OrtSessionConfig(**_engine_config_kwargs(config))
        case _:
            raise ValueError(f"unsupported engine {engine!r} for RAME models")


def _engine_config_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(config)
    providers = kwargs.get("execution_providers")
    if providers is None:
        return kwargs
    if not isinstance(providers, list):
        raise TypeError("engine_config.execution_providers must be a list")

    kwargs["execution_providers"] = [
        _execution_provider_config(provider) for provider in providers
    ]
    return kwargs


def _execution_provider_config(provider: object) -> object:
    if not isinstance(provider, dict):
        raise TypeError("engine_config.execution_providers entries must be mappings")

    provider_kwargs = dict(provider)
    kind = provider_kwargs.pop("kind", None)
    match kind:
        case "cpu":
            return CpuExecutionProviderConfig(**provider_kwargs)
        case "cuda":
            return CudaExecutionProviderConfig(**provider_kwargs)
        case "tensorrt":
            return TensorRtExecutionProviderConfig(**provider_kwargs)
        case _:
            raise ValueError(f"unsupported ONNX Runtime execution provider {kind!r}")
