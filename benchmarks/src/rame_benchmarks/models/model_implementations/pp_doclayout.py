from __future__ import annotations

from functools import partial

from rame_benchmarks.models.model_loader import ModelLoader
from rame_benchmarks.models.model_meta import ModelMeta
from rame_benchmarks.models.paddle import PaddleLayoutDetectionModel
from rame_benchmarks.models.rame import RameLayoutDetectionModel

rame_pp_doclayout_plus_onnx = ModelLoader(
    model_meta=ModelMeta(
        name="rame-pp-doclayout-plus-onnx",
        description="rame PP-DocLayout Plus ONNX implementation.",
    ),
    loader=partial(
        RameLayoutDetectionModel,
        source="PaddlePaddle/PP-DocLayout_plus-L_onnx",
        engine="onnxruntime",
    ),
    loader_kwargs={
        "engine_config": {
            "intra_op_num_threads": 10,
        },
    },
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

paddlex_pp_doclayout_plus_hpi_auto_cpu = ModelLoader(
    model_meta=ModelMeta(
        name="paddlex-pp-doclayout-plus-hpi-auto-cpu",
        description="PaddleX PP-DocLayout Plus CPU implementation using HPI auto selection.",
    ),
    loader=partial(
        PaddleLayoutDetectionModel,
        model_name="PP-DocLayout_plus-L",
        device="cpu",
        engine="hpi",
    ),
)

paddlex_pp_doclayout_plus_hpi_onnxruntime_cpu = ModelLoader(
    model_meta=ModelMeta(
        name="paddlex-pp-doclayout-plus-hpi-onnxruntime-cpu",
        description="PaddleX PP-DocLayout Plus CPU implementation using HPI ONNXRuntime.",
    ),
    loader=partial(
        PaddleLayoutDetectionModel,
        model_name="PP-DocLayout_plus-L",
        device="cpu",
        engine="hpi",
    ),
    loader_kwargs={
        "engine_config": {
            "auto_config": False,
            "backend": "onnxruntime",
            "backend_config": {
                "cpu_num_threads": 10,
            },
        },
    },
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
