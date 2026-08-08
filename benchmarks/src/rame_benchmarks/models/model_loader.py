from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from omegaconf import OmegaConf

from rame_benchmarks.models.model_meta import ModelMeta
from rame_benchmarks.models.models_protocols import BenchmarkModel


@dataclass(frozen=True)
class ModelLoader:
    model_meta: ModelMeta
    loader: Callable[..., BenchmarkModel] | None = None
    # Overrideable kwargs for the loader function.
    loader_kwargs: dict[str, Any] | None = None

    def load_model(self, **kwargs: Any) -> BenchmarkModel:
        if self.loader is None:
            raise ValueError(f"model is not implemented: {self.model_meta.name}")

        loader_kwargs = OmegaConf.merge(
            OmegaConf.create(self.loader_kwargs or {}),
            OmegaConf.create(kwargs),
        )
        return self.loader(
            self.model_meta,
            **cast(
                dict[str, Any],
                OmegaConf.to_container(loader_kwargs, resolve=True),
            ),
        )
