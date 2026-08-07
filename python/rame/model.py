from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

from rame.layout import LayoutResult


class LayoutModel(Protocol):
    """Interface implemented by all rame layout detection models."""

    def detect_layout(self, image: NDArray[np.uint8]) -> LayoutResult: ...
    def detect_layout_many(
        self, images: list[NDArray[np.uint8]]
    ) -> list[LayoutResult]: ...


__all__ = ["LayoutModel"]
