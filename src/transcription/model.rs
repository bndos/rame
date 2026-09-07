use crate::RameResult;
use crate::audio::AudioView;
use crate::runtime::expect_one;

use super::{TranscriptionOptions, TranscriptionResult};

#[derive(Debug, Clone)]
pub struct TranscriptionInput<'a> {
    pub audio: AudioView<'a>,
    pub options: TranscriptionOptions,
}

impl<'a> TranscriptionInput<'a> {
    pub fn new(audio: AudioView<'a>) -> Self {
        Self {
            audio,
            options: TranscriptionOptions::default(),
        }
    }

    pub fn with_options(mut self, options: TranscriptionOptions) -> Self {
        self.options = options;
        self
    }
}

pub trait TranscriptionModel {
    fn transcribe_many(
        &mut self,
        inputs: &[TranscriptionInput<'_>],
    ) -> RameResult<Vec<TranscriptionResult>>;

    fn transcribe(&mut self, input: TranscriptionInput<'_>) -> RameResult<TranscriptionResult> {
        let results = self.transcribe_many(std::slice::from_ref(&input))?;
        expect_one(results, "transcription output")
    }
}

#[cfg(test)]
mod tests {
    use crate::RameResult;
    use crate::audio::AudioView;
    use crate::transcription::{TimestampGranularity, TranscriptionOptions};

    use super::{TranscriptionInput, TranscriptionModel, TranscriptionResult};

    struct DurationModel;

    impl TranscriptionModel for DurationModel {
        fn transcribe_many(
            &mut self,
            inputs: &[TranscriptionInput<'_>],
        ) -> RameResult<Vec<TranscriptionResult>> {
            Ok(inputs
                .iter()
                .map(|input| TranscriptionResult::empty(input.audio.duration()))
                .collect())
        }
    }

    #[test]
    fn models_receive_the_semantic_transcription_api() {
        let samples = vec![0.0; 16_000];
        let audio = AudioView::from_interleaved_f32(16_000, 1, samples.as_slice()).unwrap();
        let mut model = DurationModel;

        let result = model.transcribe(TranscriptionInput::new(audio)).unwrap();

        assert_eq!(result.audio_duration.as_secs(), 1);
        assert!(result.results.is_empty());
    }

    #[test]
    fn default_options_request_no_optional_features() {
        assert_eq!(
            TranscriptionOptions::default().timestamps,
            TimestampGranularity::None
        );
        assert!(TranscriptionOptions::default().languages.is_empty());
        assert_eq!(TranscriptionOptions::default().max_alternatives.get(), 1);
        assert!(!TranscriptionOptions::default().word_confidence);
        assert!(TranscriptionOptions::default().diarization.is_none());
        assert_eq!(
            TranscriptionOptions::default().channel_mode,
            crate::transcription::AudioChannelMode::Automatic
        );
    }
}
