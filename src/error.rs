use thiserror::Error;

use crate::image::ImageError;
use crate::sources::SourceError;

pub type RameResult<T> = Result<T, RameError>;

#[derive(Debug, Error)]
pub enum RameError {
    #[error(transparent)]
    Image(#[from] ImageError),

    #[error(transparent)]
    Source(#[from] SourceError),
}
