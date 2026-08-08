use pyo3::prelude::*;
use rame::session::ort::OrtSessionConfig;

/// ONNX Runtime session configuration.
#[pyclass(name = "OrtSessionConfig", get_all, skip_from_py_object)]
#[derive(Debug, Clone, Default)]
pub(crate) struct PyOrtSessionConfig {
    pub(crate) intra_op_num_threads: Option<usize>,
    pub(crate) inter_op_num_threads: Option<usize>,
}

#[pymethods]
impl PyOrtSessionConfig {
    #[new]
    #[pyo3(signature = (*, intra_op_num_threads=None, inter_op_num_threads=None))]
    fn new(intra_op_num_threads: Option<usize>, inter_op_num_threads: Option<usize>) -> Self {
        Self {
            intra_op_num_threads,
            inter_op_num_threads,
        }
    }
}

impl From<&PyOrtSessionConfig> for OrtSessionConfig {
    fn from(config: &PyOrtSessionConfig) -> Self {
        let mut session_config = Self::default();

        if let Some(threads) = config.intra_op_num_threads {
            session_config = session_config.intra_threads(threads);
        }
        if let Some(threads) = config.inter_op_num_threads {
            session_config = session_config.inter_threads(threads);
        }

        session_config
    }
}
