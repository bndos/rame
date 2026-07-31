from __future__ import annotations

from functools import partial

from rame_benchmarks.models.model_loader import ModelLoader
from rame_benchmarks.models.model_meta import ModelMeta
from rame_benchmarks.models.paddle import PaddleLayoutDetectionModel

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
    loader=partial(
        PaddleLayoutDetectionModel,
        model_name="PP-DocLayout_plus-L",
    ),
)

paddlex_pp_doclayout_plus_native_cpu = ModelLoader(
    model_meta=ModelMeta(
        name="paddlex-pp-doclayout-plus-native-cpu",
        description="PaddleX PP-DocLayout Plus native CPU implementation.",
    ),
    loader=partial(
        PaddleLayoutDetectionModel,
        model_name="PP-DocLayout_plus-L",
        device="cpu",
        engine="paddle",
    ),
)

paddle_pp_doclayout_v3 = ModelLoader(
    model_meta=ModelMeta(
        name="paddle-pp-doclayout-v3",
        description="PaddleX PP-DocLayoutV3 implementation.",
    ),
    loader=partial(
        PaddleLayoutDetectionModel,
        model_name="PP-DocLayoutV3",
    ),
)
