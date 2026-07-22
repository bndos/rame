use thiserror::Error;

#[derive(Debug, Error)]
pub enum OrtError {
    #[error("ONNX Runtime error")]
    Runtime {
        #[source]
        source: ort::Error,
    },

    #[error("unsupported ONNX tensor type `{tensor_type}` for `{name}`")]
    UnsupportedTensorType { name: String, tensor_type: String },
}

impl From<ort::Error> for OrtError {
    fn from(source: ort::Error) -> Self {
        Self::Runtime { source }
    }
}

impl From<ort::Error<ort::session::builder::SessionBuilder>> for OrtError {
    fn from(err: ort::Error<ort::session::builder::SessionBuilder>) -> Self {
        let source: ort::Error = err.into();
        Self::Runtime { source }
    }
}

impl From<OrtError> for crate::RameError {
    fn from(err: OrtError) -> Self {
        crate::session::SessionError::from(err).into()
    }
}
