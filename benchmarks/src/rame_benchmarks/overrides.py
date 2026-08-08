from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from omegaconf import OmegaConf


def parse_overrides(overrides: Sequence[str] | None) -> dict[str, Any]:
    if not overrides:
        return {}

    config = OmegaConf.from_dotlist(list(overrides))
    return cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
