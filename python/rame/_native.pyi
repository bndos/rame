from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Geometry",
    "LayoutRegion",
    "LayoutResult",
    "OrtCpuConfig",
    "OrtCudaConfig",
    "OrtSessionConfig",
    "OrtTrtConfig",
    "PpDocLayoutPlusOnnx",
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

class OrtCpuConfig:
    """CPU execution provider configuration."""

    arena_allocator: bool | None

    def __init__(self, *, arena_allocator: bool | None = None) -> None: ...

class OrtCudaConfig:
    """CUDA execution provider configuration."""

    device_id: int
    memory_limit: int | None
    arena_extend_strategy: str | None
    conv_max_workspace: bool | None
    tf32: bool | None
    prefer_nhwc: bool | None

    def __init__(
        self,
        *,
        device_id: int = 0,
        memory_limit: int | None = None,
        arena_extend_strategy: str | None = None,
        conv_max_workspace: bool | None = None,
        tf32: bool | None = None,
        prefer_nhwc: bool | None = None,
    ) -> None: ...

class OrtTrtConfig:
    """TensorRT execution provider configuration."""

    device_id: int
    fp16: bool
    max_workspace_size: int | None
    min_subgraph_size: int | None
    max_partition_iterations: int | None
    engine_cache: bool | None
    engine_cache_path: str | None
    engine_cache_prefix: str | None
    context_memory_sharing: bool | None
    timing_cache: bool | None
    timing_cache_path: str | None
    force_timing_cache: bool | None
    auxiliary_streams: int | None
    profile_min_shapes: str | None
    profile_opt_shapes: str | None
    profile_max_shapes: str | None

    def __init__(
        self,
        *,
        device_id: int = 0,
        fp16: bool = False,
        max_workspace_size: int | None = None,
        min_subgraph_size: int | None = None,
        max_partition_iterations: int | None = None,
        engine_cache: bool | None = None,
        engine_cache_path: str | None = None,
        engine_cache_prefix: str | None = None,
        context_memory_sharing: bool | None = None,
        timing_cache: bool | None = None,
        timing_cache_path: str | None = None,
        force_timing_cache: bool | None = None,
        auxiliary_streams: int | None = None,
        profile_min_shapes: str | None = None,
        profile_opt_shapes: str | None = None,
        profile_max_shapes: str | None = None,
    ) -> None: ...

class OrtSessionConfig:
    """Private native ONNX Runtime session configuration."""

    execution_providers: list[OrtCpuConfig | OrtCudaConfig | OrtTrtConfig]
    graph_optimization_level: str | None
    parallel_execution: bool | None
    memory_pattern: bool | None
    deterministic_compute: bool | None
    intra_op_num_threads: int | None
    inter_op_num_threads: int | None
    config_entries: list[tuple[str, str]]

    def __init__(
        self,
        *,
        execution_providers: list[OrtCpuConfig | OrtCudaConfig | OrtTrtConfig]
        | None = None,
        graph_optimization_level: str | None = None,
        parallel_execution: bool | None = None,
        memory_pattern: bool | None = None,
        deterministic_compute: bool | None = None,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
        config_entries: list[tuple[str, str]] = [],
    ) -> None: ...

class PpDocLayoutPlusOnnx:
    def __init__(
        self, source: str, engine_config: OrtSessionConfig | None = None
    ) -> None: ...
    def detect_layout(self, image: NDArray[np.uint8]) -> LayoutResult: ...
    def detect_layout_many(
        self, images: list[NDArray[np.uint8]]
    ) -> list[LayoutResult]: ...
