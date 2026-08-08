from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

EngineName: TypeAlias = Literal["onnxruntime"]


@dataclass(frozen=True, slots=True)
class OrtSessionConfig:
    """ONNX Runtime session configuration."""

    intra_op_num_threads: int | None = None
    """Number of threads used within individual ONNX Runtime operators."""

    inter_op_num_threads: int | None = None
    """Number of threads used to run independent ONNX Runtime operators."""

    def __post_init__(self) -> None:
        _validate_thread_count("intra_op_num_threads", self.intra_op_num_threads)
        _validate_thread_count("inter_op_num_threads", self.inter_op_num_threads)


EngineConfig: TypeAlias = OrtSessionConfig | None


def _validate_thread_count(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"ONNX Runtime session config {name!r} must be an int or None")
    if value < 0:
        raise ValueError(
            f"ONNX Runtime session config {name!r} must be greater than or equal to zero"
        )


__all__ = ["EngineConfig", "EngineName", "OrtSessionConfig"]
