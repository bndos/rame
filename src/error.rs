use thiserror::Error;

use crate::image::ImageError;

pub type RameResult<T> = Result<T, RameError>;

#[derive(Debug, Error)]
pub enum RameError {
    #[error(transparent)]
    Image(#[from] ImageError),
}
