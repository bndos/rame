use std::fmt;

use ort::ep::ExecutionProviderDispatch;
use ort::session::builder::SessionBuilder;

use crate::session::ort::OrtError;

mod cpu;

pub use cpu::OrtCpuExecutionProviderConfig;

#[cfg(feature = "onnxruntime-cuda")]
mod cuda;
#[cfg(feature = "onnxruntime-tensorrt")]
mod tensorrt;
#[cfg(feature = "onnxruntime-cuda")]
pub use cuda::{OrtArenaExtendStrategy, OrtCudaExecutionProviderConfig};
#[cfg(feature = "onnxruntime-tensorrt")]
pub use tensorrt::OrtTensorRtExecutionProviderConfig;

/// ONNX Runtime session configuration.
#[derive(Clone, Default)]
pub struct OrtSessionConfig {
    pub(super) intra_threads: Option<usize>,
    pub(super) inter_threads: Option<usize>,
    pub(super) graph_optimization_level: Option<OrtGraphOptimizationLevel>,
    pub(super) parallel_execution: Option<bool>,
    pub(super) memory_pattern: Option<bool>,
    pub(super) deterministic_compute: Option<bool>,
    pub(super) config_entries: Vec<(String, String)>,
    pub(super) execution_providers: Vec<ExecutionProviderDispatch>,
    pub(super) output_names: Vec<String>,
}

impl OrtSessionConfig {
    pub fn intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = Some(threads);
        self
    }

    pub fn inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = Some(threads);
        self
    }

    pub fn graph_optimization_level(mut self, level: OrtGraphOptimizationLevel) -> Self {
        self.graph_optimization_level = Some(level);
        self
    }

    pub fn parallel_execution(mut self, enable: bool) -> Self {
        self.parallel_execution = Some(enable);
        self
    }

    pub fn memory_pattern(mut self, enable: bool) -> Self {
        self.memory_pattern = Some(enable);
        self
    }

    pub fn deterministic_compute(mut self, enable: bool) -> Self {
        self.deterministic_compute = Some(enable);
        self
    }

    pub fn config_entry(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config_entries.push((key.into(), value.into()));
        self
    }

    pub fn output(mut self, name: impl Into<String>) -> Self {
        self.output_names.push(name.into());
        self
    }

    pub(super) fn push_execution_provider(mut self, provider: ExecutionProviderDispatch) -> Self {
        self.execution_providers.push(provider);
        self
    }

    pub(super) fn apply(self, mut builder: SessionBuilder) -> Result<SessionBuilder, OrtError> {
        if !self.execution_providers.is_empty() {
            builder = builder
                .with_execution_providers(self.execution_providers)
                .map_err(OrtError::from)?;
        }

        if let Some(level) = self.graph_optimization_level {
            builder = builder
                .with_optimization_level(level.into())
                .map_err(OrtError::from)?;
        }

        if let Some(enable) = self.parallel_execution {
            builder = builder
                .with_parallel_execution(enable)
                .map_err(OrtError::from)?;
        }

        if let Some(enable) = self.memory_pattern {
            builder = builder
                .with_memory_pattern(enable)
                .map_err(OrtError::from)?;
        }

        if let Some(enable) = self.deterministic_compute {
            builder = builder
                .with_deterministic_compute(enable)
                .map_err(OrtError::from)?;
        }

        for (key, value) in self.config_entries {
            builder = builder
                .with_config_entry(key, value)
                .map_err(OrtError::from)?;
        }

        if let Some(threads) = self.intra_threads {
            builder = builder
                .with_intra_threads(threads)
                .map_err(OrtError::from)?;
        }

        if let Some(threads) = self.inter_threads {
            builder = builder
                .with_inter_threads(threads)
                .map_err(OrtError::from)?;
        }

        Ok(builder)
    }
}

impl fmt::Debug for OrtSessionConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OrtSessionConfig")
            .field("intra_threads", &self.intra_threads)
            .field("inter_threads", &self.inter_threads)
            .field("graph_optimization_level", &self.graph_optimization_level)
            .field("parallel_execution", &self.parallel_execution)
            .field("memory_pattern", &self.memory_pattern)
            .field("deterministic_compute", &self.deterministic_compute)
            .field("config_entries", &self.config_entries)
            .field("execution_providers", &self.execution_providers.len())
            .field("output_names", &self.output_names)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtGraphOptimizationLevel {
    Disable,
    Level1,
    Level2,
    Level3,
    All,
}

impl From<OrtGraphOptimizationLevel> for ort::session::builder::GraphOptimizationLevel {
    fn from(level: OrtGraphOptimizationLevel) -> Self {
        match level {
            OrtGraphOptimizationLevel::Disable => Self::Disable,
            OrtGraphOptimizationLevel::Level1 => Self::Level1,
            OrtGraphOptimizationLevel::Level2 => Self::Level2,
            OrtGraphOptimizationLevel::Level3 => Self::Level3,
            OrtGraphOptimizationLevel::All => Self::All,
        }
    }
}
