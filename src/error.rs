use std::error::Error;

use thiserror::Error;

use crate::audio::AudioError;
use crate::image::ImageError;
use crate::preprocess::PreprocessError;
use crate::runtime::RuntimeError;
use crate::session::SessionError;
use crate::sources::SourceError;
use crate::tensor::TensorError;
use crate::tokenization::TokenizationError;
use crate::transcription::TranscriptionError;

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
    Audio(#[from] AudioError),

    #[error(transparent)]
    Image(#[from] ImageError),

    #[error("model error: {0}")]
    Model(#[source] Box<dyn Error + Send + Sync>),

    #[error(transparent)]
    Preprocess(#[from] PreprocessError),

    #[error("model registry error: {0}")]
    Registry(#[source] Box<dyn Error + Send + Sync>),

    #[error(transparent)]
    Runtime(#[from] RuntimeError),

    #[error(transparent)]
    Session(#[from] SessionError),

    #[error(transparent)]
    Source(#[from] SourceError),

    #[error(transparent)]
    Tensor(#[from] TensorError),

    #[error(transparent)]
    Tokenization(#[from] TokenizationError),

    #[error(transparent)]
    Transcription(#[from] TranscriptionError),
}
