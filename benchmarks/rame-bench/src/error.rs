use thiserror::Error;

use crate::datasets::DatasetError;

pub type BenchResult<T> = Result<T, BenchError>;

#[derive(Debug, Error)]
pub enum BenchError {
    #[error(transparent)]
    Dataset(#[from] DatasetError),

    #[error("batch_size must be greater than zero")]
    InvalidBatchSize,

    #[error("image dataset did not contain any supported image files")]
    EmptyDataset,
}
