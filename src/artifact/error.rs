use std::path::PathBuf;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ArtifactError {
    #[error("artifact path must be relative: {0}")]
    AbsolutePath(PathBuf),

    #[error("artifact path must not contain parent directory components: {0}")]
    ParentDir(PathBuf),
}
