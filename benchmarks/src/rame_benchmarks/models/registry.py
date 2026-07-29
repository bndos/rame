from __future__ import annotations

from typing import Any

from rame_benchmarks.models.model_implementations import MODEL_REGISTRY
from rame_benchmarks.models.model_loader import ModelLoader
from rame_benchmarks.models.model_meta import ModelMeta, ModelName
from rame_benchmarks.models.models_protocols import BenchmarkModel


def get_model_loader(name: ModelName) -> ModelLoader:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as err:
        raise ValueError(f"unsupported model: {name}") from err


def get_model_meta(name: ModelName) -> ModelMeta:
    return get_model_loader(name).model_meta


def get_model_metas(
    names: tuple[ModelName, ...] | None = None,
) -> list[ModelMeta]:
    if names is None:
        return [loader.model_meta for loader in MODEL_REGISTRY.values()]

    return [get_model_meta(name) for name in names]


def get_model(name: ModelName, **kwargs: Any) -> BenchmarkModel:
    return get_model_loader(name).load_model(**kwargs)
