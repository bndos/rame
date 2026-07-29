from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def chunked(items: Sequence[T], size: int) -> list[Sequence[T]]:
    return [items[start : start + size] for start in range(0, len(items), size)]
