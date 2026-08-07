from __future__ import annotations

from typing import Literal

from rame import _native
from rame.model import LayoutModel


def load_layout_model(
    source: str,
    *,
    model: Literal["pp-doclayout-plus"],
    engine: Literal["onnx"] = "onnx",
) -> LayoutModel:
    match model, engine:
        case "pp-doclayout-plus", "onnx":
            return _native.PpDocLayoutPlusOnnx(source)
