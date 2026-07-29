from __future__ import annotations

from rame_benchmarks.models.model_meta import ModelMeta, ModelName
from rame_benchmarks.models.models_protocols import BenchmarkModel

MODEL_REGISTRY: tuple[ModelMeta[BenchmarkModel], ...] = (
    ModelMeta(
        name=ModelName.RAME_PP_DOCLAYOUT_PLUS_ONNX,
        description="rame PP-DocLayout Plus ONNX implementation.",
    ),
    ModelMeta(
        name=ModelName.PADDLE_PP_DOCLAYOUT_PLUS,
        description="PaddleX PP-DocLayout Plus implementation.",
    ),
)


def get_model_meta(name: ModelName) -> ModelMeta[BenchmarkModel]:
    for model in MODEL_REGISTRY:
        if model.name == name:
            return model

    raise ValueError(f"unsupported model: {name.value}")


def get_model_metas(
    names: tuple[ModelName, ...] | None = None,
) -> list[ModelMeta[BenchmarkModel]]:
    if names is None:
        return list(MODEL_REGISTRY)

    return [get_model_meta(name) for name in names]
