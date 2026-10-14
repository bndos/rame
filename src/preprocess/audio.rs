pub use rame_core::preprocess::audio::{LogMelSpectrogram, NormalizeFeatures, PreEmphasis};

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use crate::backends::mlx::audio as mlx;

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub type AudioPipeline = crate::preprocess::pipeline::PreprocessPipeline<mlx::MlxAudioBackend>;

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub type AudioPipelineBuilder =
    crate::preprocess::pipeline::PreprocessPipelineBuilder<mlx::MlxAudioBackend>;

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub fn pipeline(spectrogram: LogMelSpectrogram) -> crate::RameResult<AudioPipelineBuilder> {
    Ok(AudioPipelineBuilder::new(mlx::MlxAudioBackend::new(
        spectrogram,
    )?))
}
