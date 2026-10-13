use std::sync::Arc;

use mlx_rs::Array;

use crate::RameResult;
use crate::audio::AudioView;
use crate::preprocess::PreprocessError;
use crate::preprocess::audio::LogMelSpectrogram;
use crate::preprocess::pipeline::PreprocessBackend;

pub struct MlxAudioBackend {
    spectrogram: LogMelSpectrogram,
    assets: Arc<SpectrogramAssets>,
}

pub(super) struct SpectrogramAssets {
    pub(super) window: Array,
    pub(super) filterbank: Array,
}

pub struct MlxAudioData<'a> {
    pub(super) stage: Stage<'a>,
    pub(super) spectrogram: LogMelSpectrogram,
    pub(super) assets: Arc<SpectrogramAssets>,
}

pub(super) enum Stage<'a> {
    Pcm(Vec<AudioView<'a>>),
    Signal(Vec<Array>),
    LogMel(Vec<Array>),
    Output(Vec<Array>),
}

impl MlxAudioBackend {
    pub fn new(spectrogram: LogMelSpectrogram) -> RameResult<Self> {
        if spectrogram.sample_rate == 0
            || spectrogram.mel_bins == 0
            || spectrogram.hop_length == 0
            || spectrogram.fft_size < 2
            || spectrogram.window_size == 0
            || spectrogram.window_size > spectrogram.fft_size
            || !spectrogram.log_guard.is_finite()
            || spectrogram.log_guard <= 0.0
            || i32::try_from(spectrogram.fft_size).is_err()
            || i32::try_from(spectrogram.mel_bins).is_err()
        {
            return Err(super::invalid("invalid log-mel spectrogram configuration"));
        }
        Ok(Self {
            spectrogram,
            assets: Arc::new(SpectrogramAssets {
                window: Array::from_slice(
                    &spectrogram.hann_window(),
                    &[spectrogram.fft_size as i32],
                ),
                filterbank: Array::from_slice(
                    &spectrogram.slaney_filterbank(),
                    &[
                        spectrogram.mel_bins as i32,
                        (spectrogram.fft_size / 2 + 1) as i32,
                    ],
                ),
            }),
        })
    }
}

impl PreprocessBackend for MlxAudioBackend {
    type Source<'a> = AudioView<'a>;
    type Data<'a> = MlxAudioData<'a>;
    type Output = Vec<Array>;

    fn input<'a>(&self, sources: &'a [Self::Source<'a>]) -> RameResult<Self::Data<'a>> {
        for source in sources {
            if source.channels() != 1 || source.sample_rate() != self.spectrogram.sample_rate {
                return Err(super::invalid(
                    "MLX audio preprocessing expects mono PCM at the configured sample rate",
                ));
            }
        }
        Ok(MlxAudioData {
            stage: Stage::Pcm(sources.to_vec()),
            spectrogram: self.spectrogram,
            assets: Arc::clone(&self.assets),
        })
    }

    fn finish(&self, data: Self::Data<'_>) -> RameResult<Self::Output> {
        match data.stage {
            Stage::Output(features) => Ok(features),
            _ => Err(PreprocessError::MissingOutput.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MlxAudioBackend;
    use crate::preprocess::audio::LogMelSpectrogram;

    #[test]
    fn rejects_invalid_spectrogram_dimensions() {
        let mut spec = LogMelSpectrogram::new(16_000, 128, 512, 400, 160);
        spec.window_size = 513;
        assert!(MlxAudioBackend::new(spec).is_err());
    }
}
