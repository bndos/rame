use crate::preprocess::PreprocessError;

impl From<opencv::Error> for PreprocessError {
    fn from(err: opencv::Error) -> Self {
        Self::Backend {
            backend: "OpenCV",
            message: err.to_string(),
        }
    }
}
