use thiserror::Error;

use crate::image::ImageError;
use crate::models::ModelError;
use crate::preprocess::PreprocessError;
use crate::session::SessionError;
use crate::sources::SourceError;
use crate::tensor::TensorError;

pub type RameResult<T> = Result<T, RameError>;

#[derive(Debug, Error)]
pub enum RameError {
    #[error("invalid {stage} batch length: expected {expected}, got {actual}")]
    InvalidBatchLength {
        stage: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error(transparent)]
    Image(#[from] ImageError),

    #[error(transparent)]
    Model(#[from] ModelError),

    #[error(transparent)]
    Preprocess(#[from] PreprocessError),

    #[error(transparent)]
    Session(#[from] SessionError),

    #[error(transparent)]
    Source(#[from] SourceError),

    #[error(transparent)]
    Tensor(#[from] TensorError),
}
