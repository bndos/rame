use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TokenizationError {
    #[error("failed to load tokenizer from {path:?}: {message}")]
    Load { path: PathBuf, message: String },

    #[error("failed to decode token IDs: {message}")]
    Decode { message: String },
}
