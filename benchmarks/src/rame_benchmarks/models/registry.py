from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelName(str, Enum):
    RAME_PP_DOCLAYOUT_PLUS_ONNX = "rame-pp-doclayout-plus-onnx"
    PADDLE_PP_DOCLAYOUT_PLUS = "paddle-pp-doclayout-plus"


@dataclass(frozen=True)
class ModelMeta:
    name: ModelName
    description: str


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


def get_model_metas(names: tuple[ModelName, ...] | None = None) -> list[ModelMeta]:
    if names is None:
        return list(MODEL_REGISTRY)

    selected = set(names)
    return [model for model in MODEL_REGISTRY if model.name in selected]
