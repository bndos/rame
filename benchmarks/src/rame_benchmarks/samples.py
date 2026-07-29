from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass(frozen=True)
class ImageSample:
    path: Path
    rgb: NDArray[np.uint8]

    @classmethod
    def from_bytes(cls, path: Path, image_bytes: bytes) -> ImageSample:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return cls(path=path, rgb=image_to_rgb_array(image))

    @classmethod
    def cache_path(cls, file_name: str, output_dir: Path) -> Path:
        return output_dir / stable_filename(file_name)

    def as_ndarray(self) -> NDArray[np.uint8]:
        return self.rgb

    def write_original_bytes(self, image_bytes: bytes) -> None:
        if not self.path.exists():
            self.path.write_bytes(image_bytes)


def stable_filename(file_name: str) -> str:
    raw_name = Path(file_name).name
    suffix = Path(raw_name).suffix or ".img"
    digest = hashlib.sha256(raw_name.encode()).hexdigest()[:10]
    return f"{Path(raw_name).stem}-{digest}{suffix}"


def image_to_rgb_array(image: Image.Image) -> NDArray[np.uint8]:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)
