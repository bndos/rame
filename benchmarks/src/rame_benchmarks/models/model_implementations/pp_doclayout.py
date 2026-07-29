from __future__ import annotations

from rame_benchmarks.models.model_meta import ModelMeta, ModelName
from rame_benchmarks.models.models_protocols import BenchmarkModel

rame_pp_doclayout_plus_onnx: ModelMeta[BenchmarkModel] = ModelMeta(
    name=ModelName.RAME_PP_DOCLAYOUT_PLUS_ONNX,
    description="rame PP-DocLayout Plus ONNX implementation.",
)

paddle_pp_doclayout_plus: ModelMeta[BenchmarkModel] = ModelMeta(
    name=ModelName.PADDLE_PP_DOCLAYOUT_PLUS,
    description="PaddleX PP-DocLayout Plus implementation.",
)
