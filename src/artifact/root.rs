use std::path::{Component, Path, PathBuf};

use crate::artifact::ArtifactError;
use crate::RameResult;

/// Local root path for a resolved model artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactRoot {
    path: PathBuf,
}

impl ArtifactRoot {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn join(&self, path: impl AsRef<Path>) -> RameResult<PathBuf> {
        let path = path.as_ref();
        validate_artifact_path(path)?;

        Ok(self.path.join(path))
    }
}

fn validate_artifact_path(path: &Path) -> RameResult<()> {
    if path.is_absolute() {
        return Err(ArtifactError::AbsolutePath(path.to_path_buf()).into());
    }

    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(ArtifactError::ParentDir(path.to_path_buf()).into());
    }

    Ok(())
}

impl AsRef<Path> for ArtifactRoot {
    fn as_ref(&self) -> &Path {
        self.path()
    }
}

impl From<PathBuf> for ArtifactRoot {
    fn from(path: PathBuf) -> Self {
        Self::new(path)
    }
}

impl From<&Path> for ArtifactRoot {
    fn from(path: &Path) -> Self {
        Self::new(path)
    }
}

#[cfg(test)]
mod tests {
    use super::ArtifactRoot;
    use crate::artifact::ArtifactError;
    use crate::RameError;
    use std::path::PathBuf;

    #[test]
    fn joins_artifact_path_under_root() {
        let root = ArtifactRoot::new("/cache/snapshot");

        assert_eq!(
            root.join("onnx/inference.onnx").unwrap(),
            PathBuf::from("/cache/snapshot/onnx/inference.onnx")
        );
    }

    #[test]
    fn rejects_absolute_artifact_path() {
        let root = ArtifactRoot::new("/cache/snapshot");
        let err = root.join("/models/inference.onnx").unwrap_err();

        let RameError::Artifact(err) = err else {
            panic!("expected artifact error");
        };

        assert_eq!(
            err,
            ArtifactError::AbsolutePath(PathBuf::from("/models/inference.onnx"))
        );
    }

    #[test]
    fn rejects_parent_dir_components() {
        let root = ArtifactRoot::new("/cache/snapshot");
        let err = root.join("../inference.onnx").unwrap_err();

        let RameError::Artifact(err) = err else {
            panic!("expected artifact error");
        };

        assert_eq!(
            err,
            ArtifactError::ParentDir(PathBuf::from("../inference.onnx"))
        );
    }
}
