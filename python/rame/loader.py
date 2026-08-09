from __future__ import annotations

from rame import _native
from rame.engine import (
    CpuExecutionProviderConfig,
    CudaExecutionProviderConfig,
    EngineConfig,
    EngineName,
    ExecutionProviderConfig,
    OrtSessionConfig,
    TensorRtExecutionProviderConfig,
)
from rame.model import LayoutModel, LayoutModelName


def load_layout_model(
    source: str,
    *,
    model: LayoutModelName,
    engine: EngineName = "onnxruntime",
    engine_config: EngineConfig = None,
) -> LayoutModel:
    match model, engine:
        case "pp-doclayout-plus", "onnxruntime":
            return _native.PpDocLayoutPlusOnnx(
                source,
                engine_config=_native_ort_session_config(engine_config),
            )
        case _:
            raise ValueError(
                f"unsupported layout model {model!r} with engine {engine!r}"
            )


def _native_ort_session_config(
    config: OrtSessionConfig | None,
) -> _native.OrtSessionConfig | None:
    if config is None:
        return None

    return _native.OrtSessionConfig(
        execution_providers=[]
        if config.execution_providers is None
        else [
            _native_execution_provider_config(provider)
            for provider in config.execution_providers
        ],
        graph_optimization_level=config.graph_optimization_level,
        parallel_execution=config.parallel_execution,
        memory_pattern=config.memory_pattern,
        deterministic_compute=config.deterministic_compute,
        intra_op_num_threads=config.intra_op_num_threads,
        inter_op_num_threads=config.inter_op_num_threads,
        config_entries=[]
        if config.config_entries is None
        else list(config.config_entries.items()),
    )


def _native_execution_provider_config(
    provider: ExecutionProviderConfig,
) -> _native.OrtCpuConfig | _native.OrtCudaConfig | _native.OrtTrtConfig:
    match provider:
        case CpuExecutionProviderConfig():
            return _native.OrtCpuConfig(
                arena_allocator=provider.arena_allocator,
            )
        case CudaExecutionProviderConfig():
            return _native.OrtCudaConfig(
                device_id=provider.device_id,
                memory_limit=provider.memory_limit,
                arena_extend_strategy=provider.arena_extend_strategy,
                conv_max_workspace=provider.conv_max_workspace,
                tf32=provider.tf32,
                prefer_nhwc=provider.prefer_nhwc,
            )
        case TensorRtExecutionProviderConfig():
            return _native.OrtTrtConfig(
                device_id=provider.device_id,
                fp16=provider.fp16,
                max_workspace_size=provider.max_workspace_size,
                min_subgraph_size=provider.min_subgraph_size,
                max_partition_iterations=provider.max_partition_iterations,
                engine_cache=provider.engine_cache,
                engine_cache_path=provider.engine_cache_path,
                engine_cache_prefix=provider.engine_cache_prefix,
                context_memory_sharing=provider.context_memory_sharing,
                timing_cache=provider.timing_cache,
                timing_cache_path=provider.timing_cache_path,
                force_timing_cache=provider.force_timing_cache,
                auxiliary_streams=provider.auxiliary_streams,
            )
        case _:
            raise TypeError(f"unsupported execution provider config {type(provider)!r}")
