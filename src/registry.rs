//! Feature-independent identities for built-in model implementations.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelId {
    PpDocLayoutPlus,
    PpDocLayoutV3,
    ParakeetTdtV3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTask {
    Layout,
    Transcription,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelBackend {
    OnnxRuntime,
    Mlx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelKey {
    pub id: ModelId,
    pub backend: ModelBackend,
}

impl ModelKey {
    pub const fn new(id: ModelId, backend: ModelBackend) -> Self {
        Self { id, backend }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelRegistration {
    pub key: ModelKey,
    pub task: ModelTask,
}

pub const PP_DOCLAYOUT_PLUS_ONNX: ModelRegistration = ModelRegistration {
    key: ModelKey::new(ModelId::PpDocLayoutPlus, ModelBackend::OnnxRuntime),
    task: ModelTask::Layout,
};

pub const PP_DOCLAYOUT_V3_ONNX: ModelRegistration = ModelRegistration {
    key: ModelKey::new(ModelId::PpDocLayoutV3, ModelBackend::OnnxRuntime),
    task: ModelTask::Layout,
};

pub const PARAKEET_TDT_V3_MLX: ModelRegistration = ModelRegistration {
    key: ModelKey::new(ModelId::ParakeetTdtV3, ModelBackend::Mlx),
    task: ModelTask::Transcription,
};

const MODELS: &[ModelRegistration] = &[
    PP_DOCLAYOUT_PLUS_ONNX,
    PP_DOCLAYOUT_V3_ONNX,
    PARAKEET_TDT_V3_MLX,
];

/// Lists known built-in model implementations, regardless of build features.
pub fn models() -> &'static [ModelRegistration] {
    MODELS
}

#[cfg(test)]
mod tests {
    use super::models;

    #[test]
    fn registered_implementations_have_unique_keys() {
        let registered = models();
        for (index, model) in registered.iter().enumerate() {
            assert!(
                !registered[index + 1..]
                    .iter()
                    .any(|other| other.key == model.key)
            );
        }
    }
}
