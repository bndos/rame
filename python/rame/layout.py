from __future__ import annotations

from enum import StrEnum

from rame._native import Geometry, LayoutRegion, LayoutResult


class LayoutLabel(StrEnum):
    """Known semantic labels emitted by layout detection models."""

    TEXT = "text"
    TITLE = "title"
    PAGE_NUMBER = "page_number"
    ABSTRACT = "abstract"
    CONTENT = "content"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    IMAGE = "image"
    CHART = "chart"
    TABLE = "table"
    TABLE_CAPTION = "table_caption"
    HEADER = "header"
    HEADER_IMAGE = "header_image"
    FOOTER = "footer"
    FOOTER_IMAGE = "footer_image"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    FORMULA_NUMBER = "formula_number"
    ALGORITHM = "algorithm"
    REFERENCE = "reference"
    REFERENCE_CONTENT = "reference_content"
    ASIDE_TEXT = "aside_text"
    SEAL = "seal"


__all__ = ["Geometry", "LayoutLabel", "LayoutRegion", "LayoutResult"]
