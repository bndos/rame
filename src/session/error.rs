use std::error::Error;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("{backend} session error: {source}")]
    Backend {
        backend: &'static str,
        #[source]
        source: Box<dyn Error + Send + Sync>,
    },
}

impl SessionError {
    pub fn backend(backend: &'static str, source: impl Error + Send + Sync + 'static) -> Self {
        Self::Backend {
            backend,
            source: Box::new(source),
        }
    }
}
