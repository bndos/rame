use ort::ep::ExecutionProviderDispatch;

use crate::session::ort::OrtSessionConfig;

#[derive(Debug, Clone, Copy, Default)]
pub struct OrtCpuExecutionProviderConfig {
    pub arena_allocator: Option<bool>,
}

impl OrtCpuExecutionProviderConfig {
    pub fn arena_allocator(mut self, enable: bool) -> Self {
        self.arena_allocator = Some(enable);
        self
    }
}

impl From<OrtCpuExecutionProviderConfig> for ExecutionProviderDispatch {
    fn from(config: OrtCpuExecutionProviderConfig) -> Self {
        let mut provider = ort::ep::CPU::default();
        if let Some(arena_allocator) = config.arena_allocator {
            provider = provider.with_arena_allocator(arena_allocator);
        }
        provider.build().error_on_failure()
    }
}

impl OrtSessionConfig {
    pub fn cpu(self) -> Self {
        self.cpu_config(OrtCpuExecutionProviderConfig::default())
    }

    pub fn cpu_config(self, config: OrtCpuExecutionProviderConfig) -> Self {
        self.push_execution_provider(config.into())
    }
}
