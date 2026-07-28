from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DatasetName(str, Enum):
    DOCLAYOUT_PUBLAYNET = "doclayout-publaynet"


@dataclass(frozen=True)
class Dataset:
    name: DatasetName
    repo_id: str
    split: str


def supported_dataset(name: DatasetName) -> Dataset:
    match name:
        case DatasetName.DOCLAYOUT_PUBLAYNET:
            return Dataset(
                name=name,
                repo_id="creative-graphic-design/PubLayNet",
                split="test",
            )
