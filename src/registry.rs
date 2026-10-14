//! Built-in model catalog and default-loadable implementations.
//!
//! The catalog names known implementations independently of optional backends.
//! Default loading requires the relevant built-in preprocess and engine.

use thiserror::Error;

use crate::RameResult;
use crate::layout::LayoutModel;
use crate::sources::ResolveModelSource;
use crate::transcription::TranscriptionModel;

#[cfg(all(feature = "mlx", target_os = "macos"))]
use crate::models::parakeet;
#[cfg(all(feature = "onnxruntime", feature = "opencv"))]
use crate::models::pp_doclayout;
#[cfg(any(
    all(feature = "onnxruntime", feature = "opencv"),
    all(feature = "mlx", target_os = "macos")
))]
use crate::runtime::ModelLoader;

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

const MODELS: &[ModelRegistration] = &[
    ModelRegistration {
        key: ModelKey::new(ModelId::PpDocLayoutPlus, ModelBackend::OnnxRuntime),
        task: ModelTask::Layout,
    },
    ModelRegistration {
        key: ModelKey::new(ModelId::PpDocLayoutV3, ModelBackend::OnnxRuntime),
        task: ModelTask::Layout,
    },
    ModelRegistration {
        key: ModelKey::new(ModelId::ParakeetTdtV3, ModelBackend::Mlx),
        task: ModelTask::Transcription,
    },
];

const DEFAULT_MODELS: &[ModelRegistration] = &[
    #[cfg(all(feature = "onnxruntime", feature = "opencv"))]
    MODELS[0],
    #[cfg(all(feature = "onnxruntime", feature = "opencv"))]
    MODELS[1],
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    MODELS[2],
];

/// Lists known built-in model implementations, regardless of build features.
pub fn models() -> &'static [ModelRegistration] {
    MODELS
}

/// Lists implementations loadable with the registry's default configuration.
pub fn default_models() -> &'static [ModelRegistration] {
    DEFAULT_MODELS
}

/// Loads a registered layout model with its default loader configuration.
pub fn load_layout(
    key: ModelKey,
    _source: impl ResolveModelSource,
) -> RameResult<Box<dyn LayoutModel>> {
    require_task(key, ModelTask::Layout)?;
    require_default(key)?;
    match (key.id, key.backend) {
        #[cfg(all(feature = "onnxruntime", feature = "opencv"))]
        (ModelId::PpDocLayoutPlus, ModelBackend::OnnxRuntime) => Ok(Box::new(
            pp_doclayout::plus::onnx::Loader::default().load(_source)?,
        )),
        #[cfg(all(feature = "onnxruntime", feature = "opencv"))]
        (ModelId::PpDocLayoutV3, ModelBackend::OnnxRuntime) => Ok(Box::new(
            pp_doclayout::v3::onnx::Loader::default().load(_source)?,
        )),
        _ => Err(RegistryError::Unavailable(key).into()),
    }
}

/// Loads a registered transcription model with its default loader configuration.
pub fn load_transcription(
    key: ModelKey,
    _source: impl ResolveModelSource,
) -> RameResult<Box<dyn TranscriptionModel>> {
    require_task(key, ModelTask::Transcription)?;
    require_default(key)?;
    match (key.id, key.backend) {
        #[cfg(all(feature = "mlx", target_os = "macos"))]
        (ModelId::ParakeetTdtV3, ModelBackend::Mlx) => {
            Ok(Box::new(parakeet::mlx::Artifact::default().load(_source)?))
        }
        _ => Err(RegistryError::Unavailable(key).into()),
    }
}

fn require_task(key: ModelKey, expected: ModelTask) -> Result<(), RegistryError> {
    let registration = MODELS
        .iter()
        .find(|registration| registration.key == key)
        .ok_or(RegistryError::Unavailable(key))?;
    if registration.task != expected {
        return Err(RegistryError::WrongTask {
            key,
            expected,
            actual: registration.task,
        });
    }
    Ok(())
}

fn require_default(key: ModelKey) -> Result<(), RegistryError> {
    DEFAULT_MODELS
        .iter()
        .any(|model| model.key == key)
        .then_some(())
        .ok_or(RegistryError::Unavailable(key))
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RegistryError {
    #[error("model implementation {0:?} is not available in this build")]
    Unavailable(ModelKey),

    #[error("model implementation {key:?} has task {actual:?}, expected {expected:?}")]
    WrongTask {
        key: ModelKey,
        expected: ModelTask,
        actual: ModelTask,
    },
}

impl From<RegistryError> for crate::RameError {
    fn from(error: RegistryError) -> Self {
        Self::Registry(Box::new(error))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ModelBackend, ModelId, ModelKey, ModelTask, RegistryError, default_models, models,
        require_default, require_task,
    };

    #[test]
    fn registered_implementations_have_unique_keys() {
        let registered = models();
        for (index, model) in registered.iter().enumerate() {
            assert!(
                !registered[index + 1..]
                    .iter()
                    .any(|other| other.key == model.key)
            );
            assert!(require_task(model.key, model.task).is_ok());
        }
        assert!(
            default_models()
                .iter()
                .all(|default| registered.contains(default))
        );
    }

    #[test]
    fn unavailable_models_are_reported_without_loading() {
        let key = ModelKey::new(ModelId::ParakeetTdtV3, ModelBackend::Mlx);
        if !default_models().iter().any(|model| model.key == key) {
            assert_eq!(require_default(key), Err(RegistryError::Unavailable(key)));
        }
    }

    #[cfg(all(feature = "onnxruntime", not(feature = "opencv")))]
    #[test]
    fn onnx_models_do_not_require_opencv_to_be_known() {
        let key = ModelKey::new(ModelId::PpDocLayoutPlus, ModelBackend::OnnxRuntime);
        assert!(models().iter().any(|model| model.key == key));
        assert!(!default_models().iter().any(|model| model.key == key));
    }

    #[test]
    fn loading_with_the_wrong_task_is_rejected() {
        if let Some(model) = models().first() {
            let expected = match model.task {
                ModelTask::Layout => ModelTask::Transcription,
                ModelTask::Transcription => ModelTask::Layout,
            };
            assert_eq!(
                require_task(model.key, expected),
                Err(RegistryError::WrongTask {
                    key: model.key,
                    expected,
                    actual: model.task,
                })
            );
        }
    }
}
