use std::path::PathBuf;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SourceError {
    #[error("source artifact path must be relative: {0}")]
    AbsoluteArtifactPath(PathBuf),

    #[error("hugging face source error: {0}")]
    HuggingFace(String),

    #[error("source artifact path must not contain parent directory components: {0}")]
    ParentDirArtifactPath(PathBuf),
}
