use std::path::{Path, PathBuf};

use crate::RameResult;
use crate::sources::ResolvedModelSource;

/// Resolves a model source into a local filesystem path.
pub trait ResolveModelSource {
    /// Returns a local path for this source.
    fn resolve_model_source(&self) -> RameResult<ResolvedModelSource>;
}

impl ResolveModelSource for PathBuf {
    fn resolve_model_source(&self) -> RameResult<ResolvedModelSource> {
        Ok(ResolvedModelSource::new(self.clone()))
    }
}

impl ResolveModelSource for Path {
    fn resolve_model_source(&self) -> RameResult<ResolvedModelSource> {
        Ok(ResolvedModelSource::new(self))
    }
}

impl ResolveModelSource for &Path {
    fn resolve_model_source(&self) -> RameResult<ResolvedModelSource> {
        Ok(ResolvedModelSource::new(*self))
    }
}

impl ResolveModelSource for String {
    fn resolve_model_source(&self) -> RameResult<ResolvedModelSource> {
        Ok(ResolvedModelSource::new(self))
    }
}

impl ResolveModelSource for str {
    fn resolve_model_source(&self) -> RameResult<ResolvedModelSource> {
        Ok(ResolvedModelSource::new(self))
    }
}

impl ResolveModelSource for &str {
    fn resolve_model_source(&self) -> RameResult<ResolvedModelSource> {
        Ok(ResolvedModelSource::new(self))
    }
}

#[cfg(test)]
mod tests {
    use super::ResolveModelSource;
    use std::path::{Path, PathBuf};

    #[test]
    fn path_buf_resolves_to_itself() {
        let path = PathBuf::from("./models");

        assert_eq!(path.resolve_model_source().unwrap().path(), path);
    }

    #[test]
    fn path_ref_resolves_to_itself() {
        let path = Path::new("./models");

        assert_eq!(
            path.resolve_model_source().unwrap().path(),
            PathBuf::from("./models")
        );
    }

    #[test]
    fn string_resolves_as_local_path() {
        let path = "./models".to_string();

        assert_eq!(
            path.resolve_model_source().unwrap().path(),
            PathBuf::from("./models")
        );
    }

    #[test]
    fn str_ref_resolves_as_local_path() {
        let path = "./models";

        assert_eq!(
            path.resolve_model_source().unwrap().path(),
            PathBuf::from("./models")
        );
    }
}
