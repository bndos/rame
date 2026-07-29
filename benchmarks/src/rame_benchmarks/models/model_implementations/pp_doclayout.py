from __future__ import annotations

from rame_benchmarks.models.model_loader import ModelLoader
from rame_benchmarks.models.model_meta import ModelMeta

rame_pp_doclayout_plus_onnx = ModelLoader(
    model_meta=ModelMeta(
        name="rame-pp-doclayout-plus-onnx",
        description="rame PP-DocLayout Plus ONNX implementation.",
    )
)

paddle_pp_doclayout_plus = ModelLoader(
    model_meta=ModelMeta(
        name="paddle-pp-doclayout-plus",
        description="PaddleX PP-DocLayout Plus implementation.",
    ),
)
