from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DatasetName(str, Enum):
    DOCLAYOUT_PUBLAYNET = "doclayout-publaynet"


@dataclass(frozen=True)
class BenchmarkDataset:
    name: DatasetName
    repo_id: str
    split: str


DOCLAYOUT_PUBLAYNET = BenchmarkDataset(
    name=DatasetName.DOCLAYOUT_PUBLAYNET,
    repo_id="creative-graphic-design/PubLayNet",
    split="test",
)
