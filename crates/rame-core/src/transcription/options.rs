use std::num::NonZeroU16;

use super::TimestampGranularity;

/// How a model should handle audio containing multiple channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum AudioChannelMode {
    /// Let the model select its supported channel behavior.
    #[default]
    Automatic,
    /// Recognize all channels as one combined stream.
    Mixed,
    /// Recognize each channel independently.
    Separate,
}

/// Optional speaker-count hints for diarization.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct SpeakerDiarizationOptions {
    pub min_speakers: Option<NonZeroU16>,
    pub max_speakers: Option<NonZeroU16>,
}

/// Model-independent transcription features requested for one input.
///
/// A model must return an error rather than silently ignore a requested
/// feature it cannot support.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TranscriptionOptions {
    /// Candidate language codes ordered by preference.
    ///
    /// An empty list lets the model use its default or automatic detection.
    pub languages: Vec<String>,
    pub timestamps: TimestampGranularity,
    pub word_confidence: bool,
    /// Maximum number of ordered hypotheses requested for each result.
    pub max_alternatives: NonZeroU16,
    /// Enable speaker diarization, optionally with speaker-count hints.
    pub diarization: Option<SpeakerDiarizationOptions>,
    pub channel_mode: AudioChannelMode,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            languages: Vec::new(),
            timestamps: TimestampGranularity::None,
            word_confidence: false,
            max_alternatives: NonZeroU16::MIN,
            diarization: None,
            channel_mode: AudioChannelMode::Automatic,
        }
    }
}
