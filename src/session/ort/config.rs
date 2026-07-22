use ort::session::builder::SessionBuilder;

use crate::session::ort::OrtError;

/// ONNX Runtime session configuration.
#[derive(Debug, Clone, Default)]
pub struct OrtSessionConfig {
    pub(super) intra_threads: Option<usize>,
    pub(super) inter_threads: Option<usize>,
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

    pub(super) fn apply(self, mut builder: SessionBuilder) -> Result<SessionBuilder, OrtError> {
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
