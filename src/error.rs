use thiserror::Error;

use crate::image::ImageError;
use crate::models::ModelError;
use crate::preprocess::PreprocessError;
use crate::session::SessionError;
use crate::sources::SourceError;

pub type RameResult<T> = Result<T, RameError>;

#[derive(Debug, Error)]
pub enum RameError {
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
}
