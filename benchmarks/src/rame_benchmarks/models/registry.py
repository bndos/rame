from __future__ import annotations

from typing import Any

from rame_benchmarks.models.model_implementations import MODEL_REGISTRY
from rame_benchmarks.models.model_meta import ModelMeta, ModelName
from rame_benchmarks.models.models_protocols import BenchmarkModel


def get_model_meta(name: ModelName) -> ModelMeta[BenchmarkModel]:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as err:
        raise ValueError(f"unsupported model: {name.value}") from err


def get_model_metas(
    names: tuple[ModelName, ...] | None = None,
) -> list[ModelMeta[BenchmarkModel]]:
    if names is None:
        return list(MODEL_REGISTRY.values())

    return [get_model_meta(name) for name in names]


def get_model(name: ModelName, **kwargs: Any) -> BenchmarkModel:
    meta = get_model_meta(name)
    return meta.load_model(**kwargs)
