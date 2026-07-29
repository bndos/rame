from __future__ import annotations

from dataclasses import dataclass

ModelName = str


@dataclass(frozen=True)
class ModelMeta:
    name: ModelName
    description: str
