use pyo3::prelude::*;
use rame::session::ort::{OrtSessionConfig, OrtTensorRtExecutionProviderConfig};

use super::PyOrtTrtConfig;

impl TryFrom<&PyOrtTrtConfig> for OrtTensorRtExecutionProviderConfig {
    type Error = PyErr;

    fn try_from(config: &PyOrtTrtConfig) -> Result<Self, Self::Error> {
        let mut ep = OrtTensorRtExecutionProviderConfig::default()
            .device_id(config.device_id)
            .fp16(config.fp16);
        if let Some(size) = config.max_workspace_size {
            ep = ep.max_workspace_size(size);
        }
        if let Some(size) = config.min_subgraph_size {
            ep = ep.min_subgraph_size(size);
        }
        if let Some(iterations) = config.max_partition_iterations {
            ep = ep.max_partition_iterations(iterations);
        }
        if let Some(enable) = config.engine_cache {
            ep = ep.engine_cache(enable);
        }
        if let Some(path) = &config.engine_cache_path {
            ep = ep.engine_cache_path(path.clone());
        }
        if let Some(prefix) = &config.engine_cache_prefix {
            ep = ep.engine_cache_prefix(prefix.clone());
        }
        if let Some(enable) = config.context_memory_sharing {
            ep = ep.context_memory_sharing(enable);
        }
        if let Some(enable) = config.timing_cache {
            ep = ep.timing_cache(enable);
        }
        if let Some(path) = &config.timing_cache_path {
            ep = ep.timing_cache_path(path.clone());
        }
        if let Some(enable) = config.force_timing_cache {
            ep = ep.force_timing_cache(enable);
        }
        if let Some(streams) = config.auxiliary_streams {
            ep = ep.auxiliary_streams(streams);
        }
        if let Some(shapes) = &config.profile_min_shapes {
            ep = ep.profile_min_shapes(shapes.clone());
        }
        if let Some(shapes) = &config.profile_opt_shapes {
            ep = ep.profile_opt_shapes(shapes.clone());
        }
        if let Some(shapes) = &config.profile_max_shapes {
            ep = ep.profile_max_shapes(shapes.clone());
        }
        Ok(ep)
    }
}

impl PyOrtTrtConfig {
    pub(super) fn add_to_session_config(
        &self,
        config: OrtSessionConfig,
    ) -> PyResult<OrtSessionConfig> {
        Ok(config.tensorrt_config(OrtTensorRtExecutionProviderConfig::try_from(self)?))
    }
}
