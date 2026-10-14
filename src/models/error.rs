use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("invalid model artifact: {0}")]
    InvalidArtifact(String),

    #[error("invalid model input: {0}")]
    InvalidInput(String),

    #[error("missing tensor `{0}`")]
    MissingTensor(String),

    #[error("invalid tensor shape for `{name}`: expected {expected}, got {actual:?}")]
    InvalidTensorShape {
        name: String,
        expected: String,
        actual: Vec<usize>,
    },

    #[error("invalid tensor type for `{name}`: expected {expected}, got {actual}")]
    InvalidTensorType {
        name: String,
        expected: String,
        actual: String,
    },

    #[error("unsupported model feature `{feature}`: {reason}")]
    UnsupportedFeature {
        feature: &'static str,
        reason: &'static str,
    },
}

impl From<ModelError> for crate::RameError {
    fn from(error: ModelError) -> Self {
        Self::Model(Box::new(error))
    }
}

#[cfg(test)]
mod tests {
    use super::ModelError;
    use crate::RameError;

    #[test]
    fn preserves_model_error_as_source() {
        let error = RameError::from(ModelError::InvalidArtifact("invalid checkpoint".into()));
        let RameError::Model(source) = error else {
            panic!("expected a model error");
        };
        assert!(source.downcast_ref::<ModelError>().is_some());
    }
}
