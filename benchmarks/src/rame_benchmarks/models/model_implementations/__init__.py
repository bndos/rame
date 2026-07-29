from __future__ import annotations

import importlib
from pathlib import Path

from rame_benchmarks.models.model_loader import ModelLoader
from rame_benchmarks.models.model_meta import ModelName


def get_all_model_loaders() -> dict[ModelName, ModelLoader]:
    model_loaders = []
    package_dir = Path(__file__).parent

    for file_path in package_dir.glob("*.py"):
        if file_path.name == "__init__.py":
            continue

        module = importlib.import_module(f".{file_path.stem}", package=__name__)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, ModelLoader):
                model_loaders.append(attr)

    return {loader.model_meta.name: loader for loader in model_loaders}


MODEL_REGISTRY: dict[ModelName, ModelLoader] = get_all_model_loaders()
