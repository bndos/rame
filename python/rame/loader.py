from __future__ import annotations

from rame import _native
from rame.engine import EngineConfig, EngineName
from rame.model import LayoutModel, LayoutModelName


def load_layout_model(
    source: str,
    *,
    model: LayoutModelName,
    engine: EngineName = "onnxruntime",
    engine_config: EngineConfig = None,
) -> LayoutModel:
    match model, engine:
        case "pp-doclayout-plus", "onnxruntime":
            return _native.PpDocLayoutPlusOnnx(
                source,
                engine_config=None
                if engine_config is None
                else _native.OrtSessionConfig(
                    intra_op_num_threads=engine_config.intra_op_num_threads,
                    inter_op_num_threads=engine_config.inter_op_num_threads,
                ),
            )
        case _:
            raise ValueError(
                f"unsupported layout model {model!r} with engine {engine!r}"
            )
