use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("failed to walk image dataset at {root}")]
    Walk {
        root: PathBuf,
        #[source]
        source: walkdir::Error,
    },
}
