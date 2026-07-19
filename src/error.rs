use thiserror::Error;

use crate::artifact::ArtifactError;
use crate::image::ImageError;
use crate::sources::SourceError;

pub type RameResult<T> = Result<T, RameError>;

#[derive(Debug, Error)]
pub enum RameError {
    #[error(transparent)]
    Artifact(#[from] ArtifactError),

    #[error(transparent)]
    Image(#[from] ImageError),

    #[error(transparent)]
    Source(#[from] SourceError),
}
