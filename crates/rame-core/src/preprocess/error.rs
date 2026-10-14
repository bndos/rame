use thiserror::Error;

#[derive(Debug, Error)]
pub enum PreprocessError {
    #[error("preprocess pipeline did not produce an output")]
    MissingOutput,

    #[error("invalid preprocess tensor shape for `{name}`: expected {expected}, got {actual:?}")]
    InvalidTensorShape {
        name: &'static str,
        expected: String,
        actual: Vec<usize>,
    },

    #[error("{backend} preprocessing failed: {message}")]
    Backend {
        backend: &'static str,
        message: String,
    },

    #[error("{backend} preprocessing does not support `{op}`")]
    UnsupportedBackendOp {
        backend: &'static str,
        op: &'static str,
    },
}
