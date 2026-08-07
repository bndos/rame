from __future__ import annotations

from typing import Literal, TypeAlias

__all__ = [
    "__version__",
    "build_info",
    "Geometry",
    "LayoutRegion",
    "LayoutResult",
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
