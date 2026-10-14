use std::error::Error;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("invalid runtime input: {0}")]
    InvalidInput(String),

    #[error("invalid tensor shape for `{name}`: expected {expected}, got {actual:?}")]
    InvalidTensorShape {
        name: String,
        expected: String,
        actual: Vec<usize>,
    },

    #[error("unsupported runtime feature `{feature}`: {reason}")]
    UnsupportedFeature {
        feature: &'static str,
        reason: &'static str,
    },

    #[error("{backend} {operation} failed: {source}")]
    Backend {
        backend: &'static str,
        operation: &'static str,
        #[source]
        source: Box<dyn Error + Send + Sync>,
    },
}

impl RuntimeError {
    pub fn backend(
        backend: &'static str,
        operation: &'static str,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self::Backend {
            backend,
            operation,
            source: Box::new(source),
        }
    }
}
