from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rame_benchmarks.models.models_protocols import BenchmarkModel


class ModelName(str, Enum):
    RAME_PP_DOCLAYOUT_PLUS_ONNX = "rame-pp-doclayout-plus-onnx"
    PADDLE_PP_DOCLAYOUT_PLUS = "paddle-pp-doclayout-plus"


@dataclass(frozen=True)
class ModelMeta:
    name: ModelName
    description: str
    loader: Callable[..., BenchmarkModel] | None = None
    loader_kwargs: dict[str, Any] | None = None

    def load_model(self, **kwargs: Any) -> BenchmarkModel:
        if self.loader is None:
            raise ValueError(f"model is not implemented: {self.name.value}")

        loader_kwargs = self.loader_kwargs or {}
        return self.loader(**loader_kwargs, **kwargs)


MODEL_REGISTRY: tuple[ModelMeta, ...] = (
    ModelMeta(
        name=ModelName.RAME_PP_DOCLAYOUT_PLUS_ONNX,
        description="rame PP-DocLayout Plus ONNX implementation.",
    ),
    ModelMeta(
        name=ModelName.PADDLE_PP_DOCLAYOUT_PLUS,
        description="PaddleX PP-DocLayout Plus implementation.",
    ),
)


def get_model_meta(name: ModelName) -> ModelMeta:
    for model in MODEL_REGISTRY:
        if model.name == name:
            return model

    raise ValueError(f"unsupported model: {name.value}")


def get_model_metas(names: tuple[ModelName, ...] | None = None) -> list[ModelMeta]:
    if names is None:
        return list(MODEL_REGISTRY)

    return [get_model_meta(name) for name in names]
