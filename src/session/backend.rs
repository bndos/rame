use std::path::Path;

use crate::RameResult;
use crate::session::InferSession;

/// Loads a local model source path into an inference session.
pub trait SessionBackend {
    type Config;
    type Session: InferSession;

    fn load(path: &Path, config: Self::Config) -> RameResult<Self::Session>;
}
