from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

EngineName: TypeAlias = Literal["onnxruntime"]
GraphOptimizationLevel: TypeAlias = Literal[
    "disable", "level1", "level2", "level3", "all"
]
ArenaExtendStrategy: TypeAlias = Literal["next_power_of_two", "same_as_requested"]


@dataclass(frozen=True, slots=True)
class CpuExecutionProviderConfig:
    """ONNX Runtime CPU execution provider configuration."""

    kind: Literal["cpu"] = field(default="cpu", init=False)
    arena_allocator: bool | None = None
    """Enable or disable ONNX Runtime's CPU arena allocator."""


@dataclass(frozen=True, slots=True)
class CudaExecutionProviderConfig:
    """ONNX Runtime CUDA execution provider configuration."""

    kind: Literal["cuda"] = field(default="cuda", init=False)
    device_id: int = 0
    """CUDA device ordinal used by ONNX Runtime."""

    memory_limit: int | None = None
    """CUDA arena memory limit in bytes."""

    arena_extend_strategy: ArenaExtendStrategy | None = None
    """CUDA arena growth strategy."""

    conv_max_workspace: bool | None = None
    """Allow cuDNN convolution search to use the maximum workspace."""

    tf32: bool | None = None
    """Enable TensorFloat-32 kernels where ONNX Runtime supports them."""

    prefer_nhwc: bool | None = None
    """Prefer NHWC kernels for CUDA operators that support them."""


@dataclass(frozen=True, slots=True)
class TensorRtExecutionProviderConfig:
    """ONNX Runtime TensorRT execution provider configuration."""

    kind: Literal["tensorrt"] = field(default="tensorrt", init=False)
    device_id: int = 0
    """CUDA device ordinal used by TensorRT."""

    fp16: bool = False
    """Build TensorRT engines with FP16 precision enabled."""

    max_workspace_size: int | None = None
    min_subgraph_size: int | None = None
    max_partition_iterations: int | None = None
    engine_cache: bool | None = None
    engine_cache_path: str | None = None
    engine_cache_prefix: str | None = None
    context_memory_sharing: bool | None = None
    timing_cache: bool | None = None
    timing_cache_path: str | None = None
    force_timing_cache: bool | None = None
    auxiliary_streams: int | None = None
    profile_min_shapes: str | None = None
    """TensorRT profile minimum input shapes, e.g. ``"image:1x3x800x800"``."""

    profile_opt_shapes: str | None = None
    """TensorRT profile optimal input shapes, e.g. ``"image:32x3x800x800"``."""

    profile_max_shapes: str | None = None
    """TensorRT profile maximum input shapes, e.g. ``"image:32x3x800x800"``."""


ExecutionProviderConfig: TypeAlias = (
    CpuExecutionProviderConfig
    | CudaExecutionProviderConfig
    | TensorRtExecutionProviderConfig
)


@dataclass(frozen=True, slots=True)
class OrtSessionConfig:
    """ONNX Runtime session configuration."""

    execution_providers: Sequence[ExecutionProviderConfig] | None = None
    """Ordered execution providers registered for the session."""

    graph_optimization_level: GraphOptimizationLevel | None = None
    """ONNX Runtime graph optimization level."""

    parallel_execution: bool | None = None
    """Enable ONNX Runtime parallel graph execution mode."""

    memory_pattern: bool | None = None
    """Enable ONNX Runtime memory pattern optimization."""

    deterministic_compute: bool | None = None
    """Request deterministic kernels when ONNX Runtime supports them."""

    intra_op_num_threads: int | None = None
    """Number of threads used within individual ONNX Runtime operators."""

    inter_op_num_threads: int | None = None
    """Number of threads used to run independent ONNX Runtime operators."""

    config_entries: Mapping[str, str] | None = None
    """Raw ONNX Runtime session config entries for options not modeled yet."""


EngineConfig: TypeAlias = OrtSessionConfig | None


__all__ = [
    "ArenaExtendStrategy",
    "CpuExecutionProviderConfig",
    "CudaExecutionProviderConfig",
    "EngineConfig",
    "EngineName",
    "ExecutionProviderConfig",
    "GraphOptimizationLevel",
    "OrtSessionConfig",
    "TensorRtExecutionProviderConfig",
]
