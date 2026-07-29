from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rame_benchmarks.models.model_meta import ModelMeta
from rame_benchmarks.models.models_protocols import BenchmarkModel


@dataclass(frozen=True)
class ModelLoader:
    model_meta: ModelMeta
    loader: Callable[..., BenchmarkModel] | None = None
    loader_kwargs: dict[str, Any] | None = None

    def load_model(self, **kwargs: Any) -> BenchmarkModel:
        if self.loader is None:
            raise ValueError(f"model is not implemented: {self.model_meta.name}")

        loader_kwargs = self.loader_kwargs or {}
        return self.loader(self.model_meta, **loader_kwargs, **kwargs)
