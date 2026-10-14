//! Curated default loaders for the Rame implementations selected by this crate.

use rame::layout::LayoutModel;
use rame::models::pp_doclayout;
use rame::runtime::ModelLoader;
use rame::sources::ResolveModelSource;
use rame::transcription::TranscriptionModel;
use rame::{RameError, RameResult};
use thiserror::Error;

#[cfg(target_os = "macos")]
use rame::models::parakeet;

#[cfg(target_os = "macos")]
use rame::registry::PARAKEET_TDT_V3_MLX;
pub use rame::registry::{ModelBackend, ModelId, ModelKey, ModelRegistration, ModelTask, models};
use rame::registry::{PP_DOCLAYOUT_PLUS_ONNX, PP_DOCLAYOUT_V3_ONNX};

const DEFAULT_MODELS: &[ModelRegistration] = &[
    PP_DOCLAYOUT_PLUS_ONNX,
    PP_DOCLAYOUT_V3_ONNX,
    #[cfg(target_os = "macos")]
    PARAKEET_TDT_V3_MLX,
];

/// Lists the implementations loadable with this crate's default configuration.
pub fn default_models() -> &'static [ModelRegistration] {
    DEFAULT_MODELS
}

/// Loads a layout model with its default preprocessing and session configuration.
pub fn load_layout(
    key: ModelKey,
    source: impl ResolveModelSource,
) -> RameResult<Box<dyn LayoutModel>> {
    require_task(key, ModelTask::Layout).map_err(registry_error)?;
    require_default(key).map_err(registry_error)?;
    match (key.id, key.backend) {
        (ModelId::PpDocLayoutPlus, ModelBackend::OnnxRuntime) => Ok(Box::new(
            pp_doclayout::plus::onnx::Loader::default().load(source)?,
        )),
        (ModelId::PpDocLayoutV3, ModelBackend::OnnxRuntime) => Ok(Box::new(
            pp_doclayout::v3::onnx::Loader::default().load(source)?,
        )),
        _ => Err(registry_error(RegistryError::Unavailable(key))),
    }
}

/// Loads a transcription model with its default artifact configuration.
pub fn load_transcription(
    key: ModelKey,
    _source: impl ResolveModelSource,
) -> RameResult<Box<dyn TranscriptionModel>> {
    require_task(key, ModelTask::Transcription).map_err(registry_error)?;
    require_default(key).map_err(registry_error)?;
    match (key.id, key.backend) {
        #[cfg(target_os = "macos")]
        (ModelId::ParakeetTdtV3, ModelBackend::Mlx) => {
            Ok(Box::new(parakeet::mlx::Artifact::default().load(_source)?))
        }
        _ => Err(registry_error(RegistryError::Unavailable(key))),
    }
}

fn require_task(key: ModelKey, expected: ModelTask) -> Result<(), RegistryError> {
    let registration = models()
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

fn registry_error(error: RegistryError) -> RameError {
    RameError::Registry(Box::new(error))
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

#[cfg(test)]
mod tests {
    use super::{
        ModelBackend, ModelId, ModelKey, ModelTask, RegistryError, default_models, models,
        require_default, require_task,
    };

    #[test]
    fn default_implementations_are_in_the_catalog() {
        assert!(
            default_models()
                .iter()
                .all(|default| models().contains(default))
        );
        assert!(
            default_models()
                .iter()
                .all(|model| require_default(model.key).is_ok())
        );
    }

    #[test]
    fn known_model_with_wrong_task_is_rejected() {
        let key = ModelKey::new(ModelId::PpDocLayoutPlus, ModelBackend::OnnxRuntime);
        assert_eq!(
            require_task(key, ModelTask::Transcription),
            Err(RegistryError::WrongTask {
                key,
                expected: ModelTask::Transcription,
                actual: ModelTask::Layout,
            })
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn platform_only_default_is_unavailable() {
        let key = ModelKey::new(ModelId::ParakeetTdtV3, ModelBackend::Mlx);
        assert_eq!(require_default(key), Err(RegistryError::Unavailable(key)));
    }
}
