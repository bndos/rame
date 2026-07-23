use thiserror::Error;

#[derive(Debug, Error)]
pub enum PreprocessError {
    #[error("preprocess pipeline did not produce an output")]
    MissingOutput,
}
