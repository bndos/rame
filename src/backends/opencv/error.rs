use crate::preprocess::PreprocessError;

pub(super) fn opencv_error(err: opencv::Error) -> PreprocessError {
    PreprocessError::Backend {
        backend: "OpenCV",
        message: err.to_string(),
    }
}
