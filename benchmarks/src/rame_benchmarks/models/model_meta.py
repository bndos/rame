from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

ModelT = TypeVar("ModelT")


class ModelLoader(Protocol[ModelT]):
    def __call__(self, model_meta: ModelMeta[ModelT], **kwargs: Any) -> ModelT: ...


class ModelName(str, Enum):
    RAME_PP_DOCLAYOUT_PLUS_ONNX = "rame-pp-doclayout-plus-onnx"
    PADDLE_PP_DOCLAYOUT_PLUS = "paddle-pp-doclayout-plus"


@dataclass(frozen=True)
class ModelMeta(Generic[ModelT]):
    name: ModelName
    description: str
    loader: ModelLoader[ModelT] | None = None
    loader_kwargs: dict[str, Any] | None = None

    def load_model(self, **kwargs: Any) -> ModelT:
        if self.loader is None:
            raise ValueError(f"model is not implemented: {self.name.value}")

        loader_kwargs = self.loader_kwargs or {}
        return self.loader(self, **loader_kwargs, **kwargs)
