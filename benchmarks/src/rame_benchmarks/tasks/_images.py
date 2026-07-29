from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

from PIL import Image as PILImage


def write_image(record: dict[str, Any], output_dir: Path) -> Path:
    image_bytes = record["image"]["bytes"]
    path = output_dir / stable_filename(record["file_name"], image_bytes)
    if not path.exists():
        path.write_bytes(image_bytes)
    return path


def stable_filename(file_name: str, image_bytes: bytes) -> str:
    raw_name = Path(file_name).name
    suffix = Path(raw_name).suffix
    if not suffix:
        with PILImage.open(io.BytesIO(image_bytes)) as img:
            if img.format is None:
                raise ValueError(f"Could not determine image format for {file_name!r}")
            suffix = f".{img.format.lower()}"
    digest = hashlib.sha256(raw_name.encode()).hexdigest()[:10]
    return f"{Path(raw_name).stem}-{digest}{suffix}"
