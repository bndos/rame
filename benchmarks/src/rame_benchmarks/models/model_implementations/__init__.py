from __future__ import annotations

import importlib
from pathlib import Path

from rame_benchmarks.models.model_meta import ModelMeta, ModelName
from rame_benchmarks.models.models_protocols import BenchmarkModel


def get_all_model_meta_objects() -> dict[ModelName, ModelMeta[BenchmarkModel]]:
    model_meta_objects = []
    package_dir = Path(__file__).parent

    for file_path in package_dir.glob("*.py"):
        if file_path.name == "__init__.py":
            continue

        module = importlib.import_module(f".{file_path.stem}", package=__name__)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, ModelMeta):
                model_meta_objects.append(attr)

    return {meta.name: meta for meta in model_meta_objects}


MODEL_REGISTRY: dict[ModelName, ModelMeta[BenchmarkModel]] = (
    get_all_model_meta_objects()
)
