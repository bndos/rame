use std::path::{Path, PathBuf};

use crate::RameResult;

/// Resolves a model source into a local model file path.
pub trait ResolveModelSource {
    /// Returns a local path for `artifact_file`.
    ///
    /// Local file sources may ignore `artifact_file` because they already identify the model file
    /// directly. Repository-backed sources use it as the artifact-relative file to fetch or locate.
    fn resolve_model_file(&self, artifact_file: &str) -> RameResult<PathBuf>;
}

impl ResolveModelSource for PathBuf {
    fn resolve_model_file(&self, _artifact_file: &str) -> RameResult<PathBuf> {
        Ok(self.clone())
    }
}

impl ResolveModelSource for Path {
    fn resolve_model_file(&self, _artifact_file: &str) -> RameResult<PathBuf> {
        Ok(self.to_path_buf())
    }
}

impl ResolveModelSource for &Path {
    fn resolve_model_file(&self, _artifact_file: &str) -> RameResult<PathBuf> {
        Ok((*self).to_path_buf())
    }
}

impl ResolveModelSource for String {
    fn resolve_model_file(&self, _artifact_file: &str) -> RameResult<PathBuf> {
        Ok(PathBuf::from(self))
    }
}

impl ResolveModelSource for str {
    fn resolve_model_file(&self, _artifact_file: &str) -> RameResult<PathBuf> {
        Ok(PathBuf::from(self))
    }
}

impl ResolveModelSource for &str {
    fn resolve_model_file(&self, _artifact_file: &str) -> RameResult<PathBuf> {
        Ok(PathBuf::from(self))
    }
}
