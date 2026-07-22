use thiserror::Error;

#[derive(Debug, Error)]
pub enum SessionError {
    #[cfg(feature = "onnxruntime")]
    #[error(transparent)]
    Ort(#[from] crate::session::ort::OrtError),
}
