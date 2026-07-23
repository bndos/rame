use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("missing tensor `{0}`")]
    MissingTensor(String),

    #[error("invalid tensor shape for `{name}`: expected {expected}, got {actual:?}")]
    InvalidTensorShape {
        name: String,
        expected: String,
        actual: Vec<usize>,
    },

    #[error("invalid tensor type for `{name}`: expected {expected}, got {actual}")]
    InvalidTensorType {
        name: String,
        expected: String,
        actual: String,
    },
}
