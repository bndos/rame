#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("hugging face source error: {0}")]
    HuggingFace(String),
}
