use ort::ep::ExecutionProviderDispatch;

use crate::session::ort::OrtSessionConfig;

#[derive(Debug, Clone, Copy, Default)]
pub struct OrtCudaExecutionProviderConfig {
    pub device_id: i32,
    pub memory_limit: Option<usize>,
    pub arena_extend_strategy: Option<OrtArenaExtendStrategy>,
    pub conv_max_workspace: Option<bool>,
    pub tf32: Option<bool>,
    pub prefer_nhwc: Option<bool>,
}

impl OrtCudaExecutionProviderConfig {
    pub fn device_id(mut self, device_id: i32) -> Self {
        self.device_id = device_id;
        self
    }

    pub fn memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = Some(limit);
        self
    }

    pub fn arena_extend_strategy(mut self, strategy: OrtArenaExtendStrategy) -> Self {
        self.arena_extend_strategy = Some(strategy);
        self
    }

    pub fn conv_max_workspace(mut self, enable: bool) -> Self {
        self.conv_max_workspace = Some(enable);
        self
    }

    pub fn tf32(mut self, enable: bool) -> Self {
        self.tf32 = Some(enable);
        self
    }

    pub fn prefer_nhwc(mut self, enable: bool) -> Self {
        self.prefer_nhwc = Some(enable);
        self
    }
}

impl From<OrtCudaExecutionProviderConfig> for ExecutionProviderDispatch {
    fn from(config: OrtCudaExecutionProviderConfig) -> Self {
        let mut provider = ort::ep::CUDA::default().with_device_id(config.device_id);

        if let Some(limit) = config.memory_limit {
            provider = provider.with_memory_limit(limit);
        }
        if let Some(strategy) = config.arena_extend_strategy {
            provider = provider.with_arena_extend_strategy(strategy.into());
        }
        if let Some(enable) = config.conv_max_workspace {
            provider = provider.with_conv_max_workspace(enable);
        }
        if let Some(enable) = config.tf32 {
            provider = provider.with_tf32(enable);
        }
        if let Some(enable) = config.prefer_nhwc {
            provider = provider.with_prefer_nhwc(enable);
        }

        provider.build().error_on_failure()
    }
}

impl OrtSessionConfig {
    pub fn cuda(self, device_id: i32) -> Self {
        self.cuda_config(OrtCudaExecutionProviderConfig::default().device_id(device_id))
    }

    pub fn cuda_config(self, config: OrtCudaExecutionProviderConfig) -> Self {
        self.push_execution_provider(config.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtArenaExtendStrategy {
    NextPowerOfTwo,
    SameAsRequested,
}

impl From<OrtArenaExtendStrategy> for ort::ep::ArenaExtendStrategy {
    fn from(strategy: OrtArenaExtendStrategy) -> Self {
        match strategy {
            OrtArenaExtendStrategy::NextPowerOfTwo => Self::NextPowerOfTwo,
            OrtArenaExtendStrategy::SameAsRequested => Self::SameAsRequested,
        }
    }
}
