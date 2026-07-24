use thiserror::Error;

#[derive(Debug, Error)]
pub enum PreprocessError {
    #[error("preprocess pipeline did not produce an output")]
    MissingOutput,

    #[error("{backend} preprocessing failed: {message}")]
    Backend {
        backend: &'static str,
        message: String,
    },
}
