use pyo3::PyErr;
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use rame::RameError;

pub(crate) fn into_py_err(err: RameError) -> PyErr {
    match err {
        RameError::Audio(_)
        | RameError::Image(_)
        | RameError::Transcription(_)
        | RameError::InvalidBatchLength { .. } => PyValueError::new_err(err.to_string()),
        RameError::Source(_) => PyOSError::new_err(err.to_string()),
        RameError::Session(_)
        | RameError::Model(_)
        | RameError::Preprocess(_)
        | RameError::Tensor(_) => PyRuntimeError::new_err(err.to_string()),
    }
}
