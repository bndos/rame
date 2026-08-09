use ort::ep::ExecutionProviderDispatch;

use crate::session::ort::OrtSessionConfig;

#[derive(Debug, Clone, Default)]
pub struct OrtTensorRtExecutionProviderConfig {
    pub device_id: i32,
    pub fp16: bool,
    pub max_workspace_size: Option<usize>,
    pub min_subgraph_size: Option<usize>,
    pub max_partition_iterations: Option<u32>,
    pub engine_cache: Option<bool>,
    pub engine_cache_path: Option<String>,
    pub engine_cache_prefix: Option<String>,
    pub context_memory_sharing: Option<bool>,
    pub timing_cache: Option<bool>,
    pub timing_cache_path: Option<String>,
    pub force_timing_cache: Option<bool>,
    pub auxiliary_streams: Option<i8>,
}

impl OrtTensorRtExecutionProviderConfig {
    pub fn device_id(mut self, device_id: i32) -> Self {
        self.device_id = device_id;
        self
    }

    pub fn fp16(mut self, enable: bool) -> Self {
        self.fp16 = enable;
        self
    }

    pub fn max_workspace_size(mut self, size: usize) -> Self {
        self.max_workspace_size = Some(size);
        self
    }

    pub fn min_subgraph_size(mut self, size: usize) -> Self {
        self.min_subgraph_size = Some(size);
        self
    }

    pub fn max_partition_iterations(mut self, iterations: u32) -> Self {
        self.max_partition_iterations = Some(iterations);
        self
    }

    pub fn engine_cache(mut self, enable: bool) -> Self {
        self.engine_cache = Some(enable);
        self
    }

    pub fn engine_cache_path(mut self, path: impl Into<String>) -> Self {
        self.engine_cache_path = Some(path.into());
        self
    }

    pub fn engine_cache_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.engine_cache_prefix = Some(prefix.into());
        self
    }

    pub fn context_memory_sharing(mut self, enable: bool) -> Self {
        self.context_memory_sharing = Some(enable);
        self
    }

    pub fn timing_cache(mut self, enable: bool) -> Self {
        self.timing_cache = Some(enable);
        self
    }

    pub fn timing_cache_path(mut self, path: impl Into<String>) -> Self {
        self.timing_cache_path = Some(path.into());
        self
    }

    pub fn force_timing_cache(mut self, enable: bool) -> Self {
        self.force_timing_cache = Some(enable);
        self
    }

    pub fn auxiliary_streams(mut self, streams: i8) -> Self {
        self.auxiliary_streams = Some(streams);
        self
    }
}

impl From<OrtTensorRtExecutionProviderConfig> for ExecutionProviderDispatch {
    fn from(config: OrtTensorRtExecutionProviderConfig) -> Self {
        let mut provider = ort::ep::TensorRT::default()
            .with_device_id(config.device_id)
            .with_fp16(config.fp16);

        if let Some(size) = config.max_workspace_size {
            provider = provider.with_max_workspace_size(size);
        }
        if let Some(size) = config.min_subgraph_size {
            provider = provider.with_min_subgraph_size(size);
        }
        if let Some(iterations) = config.max_partition_iterations {
            provider = provider.with_max_partition_iterations(iterations);
        }
        if let Some(enable) = config.engine_cache {
            provider = provider.with_engine_cache(enable);
        }
        if let Some(path) = &config.engine_cache_path {
            provider = provider.with_engine_cache_path(path);
        }
        if let Some(prefix) = &config.engine_cache_prefix {
            provider = provider.with_engine_cache_prefix(prefix);
        }
        if let Some(enable) = config.context_memory_sharing {
            provider = provider.with_context_memory_sharing(enable);
        }
        if let Some(enable) = config.timing_cache {
            provider = provider.with_timing_cache(enable);
        }
        if let Some(path) = &config.timing_cache_path {
            provider = provider.with_timing_cache_path(path);
        }
        if let Some(enable) = config.force_timing_cache {
            provider = provider.with_force_timing_cache(enable);
        }
        if let Some(streams) = config.auxiliary_streams {
            provider = provider.with_auxiliary_streams(streams);
        }

        provider.build().error_on_failure()
    }
}

impl OrtSessionConfig {
    pub fn tensorrt(self, device_id: i32) -> Self {
        self.tensorrt_config(OrtTensorRtExecutionProviderConfig::default().device_id(device_id))
    }

    pub fn tensorrt_fp16(self, device_id: i32) -> Self {
        self.tensorrt_config(
            OrtTensorRtExecutionProviderConfig::default()
                .device_id(device_id)
                .fp16(true),
        )
    }

    pub fn tensorrt_config(self, config: OrtTensorRtExecutionProviderConfig) -> Self {
        self.push_execution_provider(config.into())
    }
}
