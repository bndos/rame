use std::path::{Component, Path, PathBuf};

use crate::RameResult;
use crate::sources::SourceError;

/// Local filesystem root produced by resolving a model source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelSource {
    path: PathBuf,
}

impl ResolvedModelSource {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn join_artifact_path(&self, path: impl AsRef<Path>) -> RameResult<PathBuf> {
        let path = path.as_ref();
        validate_artifact_path(path)?;

        Ok(self.path.join(path))
    }
}

fn validate_artifact_path(path: &Path) -> RameResult<()> {
    if path.is_absolute() {
        return Err(SourceError::AbsoluteArtifactPath(path.to_path_buf()).into());
    }

    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(SourceError::ParentDirArtifactPath(path.to_path_buf()).into());
    }

    Ok(())
}

impl AsRef<Path> for ResolvedModelSource {
    fn as_ref(&self) -> &Path {
        self.path()
    }
}

impl From<PathBuf> for ResolvedModelSource {
    fn from(path: PathBuf) -> Self {
        Self::new(path)
    }
}

impl From<&Path> for ResolvedModelSource {
    fn from(path: &Path) -> Self {
        Self::new(path)
    }
}

#[cfg(test)]
mod tests {
    use super::ResolvedModelSource;
    use crate::RameError;
    use crate::sources::SourceError;
    use std::path::PathBuf;

    #[test]
    fn joins_artifact_path_under_source_root() {
        let source = ResolvedModelSource::new("/cache/snapshot");

        assert_eq!(
            source.join_artifact_path("onnx/inference.onnx").unwrap(),
            PathBuf::from("/cache/snapshot/onnx/inference.onnx")
        );
    }

    #[test]
    fn rejects_absolute_artifact_path() {
        let source = ResolvedModelSource::new("/cache/snapshot");
        let err = source
            .join_artifact_path("/models/inference.onnx")
            .unwrap_err();

        let RameError::Source(err) = err else {
            panic!("expected source error");
        };

        assert_eq!(
            err,
            SourceError::AbsoluteArtifactPath(PathBuf::from("/models/inference.onnx"))
        );
    }

    #[test]
    fn rejects_parent_dir_components() {
        let source = ResolvedModelSource::new("/cache/snapshot");
        let err = source.join_artifact_path("../inference.onnx").unwrap_err();

        let RameError::Source(err) = err else {
            panic!("expected source error");
        };

        assert_eq!(
            err,
            SourceError::ParentDirArtifactPath(PathBuf::from("../inference.onnx"))
        );
    }
}
