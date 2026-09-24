use super::TdtDecodingConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdtGreedySearch<E> {
    config: TdtDecodingConfig,
    executor: E,
}

impl<E> TdtGreedySearch<E> {
    pub fn with_executor(config: TdtDecodingConfig, executor: E) -> Self {
        Self { config, executor }
    }

    pub fn config(&self) -> &TdtDecodingConfig {
        &self.config
    }

    pub fn executor(&self) -> &E {
        &self.executor
    }

    pub fn executor_mut(&mut self) -> &mut E {
        &mut self.executor
    }
}
