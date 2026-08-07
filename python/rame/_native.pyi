from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Geometry",
    "LayoutRegion",
    "LayoutResult",
    "__version__",
    "build_info",
]

__version__: str
build_info: str

GeometryKind: TypeAlias = Literal["rect", "polygon"]

class Geometry:
    """Geometry describing a detected layout region."""

    kind: GeometryKind
    """Geometry kind: `rect` or `polygon`."""

    coordinates: list[float]
    """Flat image coordinates for the geometry."""

class LayoutRegion:
    """A detected document layout region."""

    label: str
    """Semantic region label as a stable snake_case string."""

    score: float
    """Model confidence score."""

    geometry: Geometry
    """Region geometry in image coordinates."""

    reading_order: int | None
    """Optional zero-based reading order if emitted by the model."""

class LayoutResult:
    """Result of a document layout detection run."""

    regions: list[LayoutRegion]
    """Detected layout regions."""

class PpDocLayoutPlusOnnx:
    def __init__(self, source: str) -> None: ...
    def detect_layout(self, image: NDArray[np.uint8]) -> LayoutResult: ...
    def detect_layout_many(
        self, images: list[NDArray[np.uint8]]
    ) -> list[LayoutResult]: ...
