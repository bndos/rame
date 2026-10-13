mod log_mel;
mod normalize;
mod preemphasis;
mod state;

pub use state::MlxAudioBackend;

use mlx_rs::error::Exception;

use crate::preprocess::PreprocessError;

fn mlx_error(error: Exception) -> crate::RameError {
    PreprocessError::Backend {
        backend: "MLX",
        message: error.to_string(),
    }
    .into()
}

fn invalid(message: &'static str) -> crate::RameError {
    PreprocessError::Backend {
        backend: "MLX",
        message: message.into(),
    }
    .into()
}
