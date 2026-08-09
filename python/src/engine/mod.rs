use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use rame::session::ort::{
    OrtCpuExecutionProviderConfig, OrtGraphOptimizationLevel, OrtSessionConfig,
};

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "tensorrt")]
mod tensorrt;

#[derive(Debug, Clone)]
pub(crate) enum PyOrtExecutionProviderConfig {
    Cpu(PyOrtCpuConfig),
    Cuda(PyOrtCudaConfig),
    Trt(PyOrtTrtConfig),
}

impl<'py> FromPyObject<'_, 'py> for PyOrtExecutionProviderConfig {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(config) = obj.cast::<PyOrtCpuConfig>() {
            return Ok(Self::Cpu(config.try_borrow()?.clone()));
        }
        if let Ok(config) = obj.cast::<PyOrtCudaConfig>() {
            return Ok(Self::Cuda(config.try_borrow()?.clone()));
        }
        if let Ok(config) = obj.cast::<PyOrtTrtConfig>() {
            return Ok(Self::Trt(config.try_borrow()?.clone()));
        }

        let type_name = obj
            .get_type()
            .qualname()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "unknown".to_string());
        Err(PyTypeError::new_err(format!(
            "unsupported execution provider type: {type_name}"
        )))
    }
}

/// CPU execution provider configuration.
#[pyclass(name = "OrtCpuConfig", get_all, skip_from_py_object)]
#[derive(Debug, Clone, Default)]
pub(crate) struct PyOrtCpuConfig {
    pub(crate) arena_allocator: Option<bool>,
}

#[pymethods]
impl PyOrtCpuConfig {
    #[new]
    #[pyo3(signature = (*, arena_allocator=None))]
    fn new(arena_allocator: Option<bool>) -> Self {
        Self { arena_allocator }
    }
}

impl From<&PyOrtCpuConfig> for OrtCpuExecutionProviderConfig {
    fn from(config: &PyOrtCpuConfig) -> Self {
        let mut ep = OrtCpuExecutionProviderConfig::default();
        if let Some(enable) = config.arena_allocator {
            ep = ep.arena_allocator(enable);
        }
        ep
    }
}

/// CUDA execution provider configuration.
#[pyclass(name = "OrtCudaConfig", get_all, skip_from_py_object)]
#[derive(Debug, Clone, Default)]
pub(crate) struct PyOrtCudaConfig {
    pub(crate) device_id: i32,
    pub(crate) memory_limit: Option<usize>,
    pub(crate) arena_extend_strategy: Option<String>,
    pub(crate) conv_max_workspace: Option<bool>,
    pub(crate) tf32: Option<bool>,
    pub(crate) prefer_nhwc: Option<bool>,
}

#[pymethods]
impl PyOrtCudaConfig {
    #[new]
    #[pyo3(signature = (
        *,
        device_id=0,
        memory_limit=None,
        arena_extend_strategy=None,
        conv_max_workspace=None,
        tf32=None,
        prefer_nhwc=None,
    ))]
    fn new(
        device_id: i32,
        memory_limit: Option<usize>,
        arena_extend_strategy: Option<String>,
        conv_max_workspace: Option<bool>,
        tf32: Option<bool>,
        prefer_nhwc: Option<bool>,
    ) -> Self {
        Self {
            device_id,
            memory_limit,
            arena_extend_strategy,
            conv_max_workspace,
            tf32,
            prefer_nhwc,
        }
    }
}

/// TensorRT execution provider configuration.
#[pyclass(name = "OrtTrtConfig", get_all, skip_from_py_object)]
#[derive(Debug, Clone, Default)]
pub(crate) struct PyOrtTrtConfig {
    pub(crate) device_id: i32,
    pub(crate) fp16: bool,
    pub(crate) max_workspace_size: Option<usize>,
    pub(crate) min_subgraph_size: Option<usize>,
    pub(crate) max_partition_iterations: Option<u32>,
    pub(crate) engine_cache: Option<bool>,
    pub(crate) engine_cache_path: Option<String>,
    pub(crate) engine_cache_prefix: Option<String>,
    pub(crate) context_memory_sharing: Option<bool>,
    pub(crate) timing_cache: Option<bool>,
    pub(crate) timing_cache_path: Option<String>,
    pub(crate) force_timing_cache: Option<bool>,
    pub(crate) auxiliary_streams: Option<i8>,
    pub(crate) profile_min_shapes: Option<String>,
    pub(crate) profile_opt_shapes: Option<String>,
    pub(crate) profile_max_shapes: Option<String>,
}

#[pymethods]
impl PyOrtTrtConfig {
    #[new]
    #[pyo3(signature = (
        *,
        device_id=0,
        fp16=false,
        max_workspace_size=None,
        min_subgraph_size=None,
        max_partition_iterations=None,
        engine_cache=None,
        engine_cache_path=None,
        engine_cache_prefix=None,
        context_memory_sharing=None,
        timing_cache=None,
        timing_cache_path=None,
        force_timing_cache=None,
        auxiliary_streams=None,
        profile_min_shapes=None,
        profile_opt_shapes=None,
        profile_max_shapes=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        device_id: i32,
        fp16: bool,
        max_workspace_size: Option<usize>,
        min_subgraph_size: Option<usize>,
        max_partition_iterations: Option<u32>,
        engine_cache: Option<bool>,
        engine_cache_path: Option<String>,
        engine_cache_prefix: Option<String>,
        context_memory_sharing: Option<bool>,
        timing_cache: Option<bool>,
        timing_cache_path: Option<String>,
        force_timing_cache: Option<bool>,
        auxiliary_streams: Option<i8>,
        profile_min_shapes: Option<String>,
        profile_opt_shapes: Option<String>,
        profile_max_shapes: Option<String>,
    ) -> Self {
        Self {
            device_id,
            fp16,
            max_workspace_size,
            min_subgraph_size,
            max_partition_iterations,
            engine_cache,
            engine_cache_path,
            engine_cache_prefix,
            context_memory_sharing,
            timing_cache,
            timing_cache_path,
            force_timing_cache,
            auxiliary_streams,
            profile_min_shapes,
            profile_opt_shapes,
            profile_max_shapes,
        }
    }
}

/// ONNX Runtime session configuration.
#[pyclass(name = "OrtSessionConfig", skip_from_py_object)]
#[derive(Debug, Default)]
pub(crate) struct PyOrtSessionConfig {
    pub(crate) execution_providers: Vec<PyOrtExecutionProviderConfig>,
    pub(crate) graph_optimization_level: Option<String>,
    pub(crate) parallel_execution: Option<bool>,
    pub(crate) memory_pattern: Option<bool>,
    pub(crate) deterministic_compute: Option<bool>,
    pub(crate) intra_op_num_threads: Option<usize>,
    pub(crate) inter_op_num_threads: Option<usize>,
    pub(crate) config_entries: Vec<(String, String)>,
}

#[pymethods]
impl PyOrtSessionConfig {
    #[new]
    #[pyo3(signature = (
        *,
        execution_providers=None,
        graph_optimization_level=None,
        parallel_execution=None,
        memory_pattern=None,
        deterministic_compute=None,
        intra_op_num_threads=None,
        inter_op_num_threads=None,
        config_entries=Vec::new(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        execution_providers: Option<Vec<PyOrtExecutionProviderConfig>>,
        graph_optimization_level: Option<String>,
        parallel_execution: Option<bool>,
        memory_pattern: Option<bool>,
        deterministic_compute: Option<bool>,
        intra_op_num_threads: Option<usize>,
        inter_op_num_threads: Option<usize>,
        config_entries: Vec<(String, String)>,
    ) -> Self {
        Self {
            execution_providers: execution_providers.unwrap_or_default(),
            graph_optimization_level,
            parallel_execution,
            memory_pattern,
            deterministic_compute,
            intra_op_num_threads,
            inter_op_num_threads,
            config_entries,
        }
    }
}

impl PyOrtSessionConfig {
    pub(crate) fn to_ort_session_config(&self) -> PyResult<OrtSessionConfig> {
        let mut session_config = OrtSessionConfig::default();

        if let Some(level) = self.graph_optimization_level.as_deref() {
            session_config =
                session_config.graph_optimization_level(parse_graph_optimization_level(level)?);
        }
        if let Some(enable) = self.parallel_execution {
            session_config = session_config.parallel_execution(enable);
        }
        if let Some(enable) = self.memory_pattern {
            session_config = session_config.memory_pattern(enable);
        }
        if let Some(enable) = self.deterministic_compute {
            session_config = session_config.deterministic_compute(enable);
        }
        if let Some(threads) = self.intra_op_num_threads {
            session_config = session_config.intra_threads(threads);
        }
        if let Some(threads) = self.inter_op_num_threads {
            session_config = session_config.inter_threads(threads);
        }
        for (key, value) in &self.config_entries {
            session_config = session_config.config_entry(key.clone(), value.clone());
        }
        for ep in &self.execution_providers {
            session_config = ep.add_to_session_config(session_config)?;
        }

        Ok(session_config)
    }
}

impl PyOrtExecutionProviderConfig {
    fn add_to_session_config(&self, config: OrtSessionConfig) -> PyResult<OrtSessionConfig> {
        match self {
            Self::Cpu(cpu) => cpu.add_to_session_config(config),
            Self::Cuda(cuda) => cuda.add_to_session_config(config),
            Self::Trt(trt) => trt.add_to_session_config(config),
        }
    }
}

impl PyOrtCpuConfig {
    fn add_to_session_config(&self, config: OrtSessionConfig) -> PyResult<OrtSessionConfig> {
        Ok(config.cpu_config(OrtCpuExecutionProviderConfig::from(self)))
    }
}

#[cfg(not(feature = "cuda"))]
impl PyOrtCudaConfig {
    fn add_to_session_config(&self, _config: OrtSessionConfig) -> PyResult<OrtSessionConfig> {
        Err(PyValueError::new_err(
            "CUDA execution provider requested, but this rame wheel was not built with CUDA support",
        ))
    }
}

#[cfg(not(feature = "tensorrt"))]
impl PyOrtTrtConfig {
    fn add_to_session_config(&self, _config: OrtSessionConfig) -> PyResult<OrtSessionConfig> {
        Err(PyValueError::new_err(
            "TensorRT execution provider requested, but this rame wheel was not built with TensorRT support",
        ))
    }
}

fn parse_graph_optimization_level(level: &str) -> PyResult<OrtGraphOptimizationLevel> {
    match level {
        "disable" => Ok(OrtGraphOptimizationLevel::Disable),
        "level1" => Ok(OrtGraphOptimizationLevel::Level1),
        "level2" => Ok(OrtGraphOptimizationLevel::Level2),
        "level3" => Ok(OrtGraphOptimizationLevel::Level3),
        "all" => Ok(OrtGraphOptimizationLevel::All),
        value => Err(PyValueError::new_err(format!(
            "unsupported ONNX Runtime graph optimization level '{value}'"
        ))),
    }
}
