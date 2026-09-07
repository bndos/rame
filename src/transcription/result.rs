use std::time::Duration;

use super::{SpeakerId, TimeSpan};

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionWord {
    pub text: String,
    /// Location in the input audio when word timestamps were requested and available.
    pub time: Option<TimeSpan>,
    pub confidence: Option<f32>,
    pub speaker: Option<SpeakerId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionAlternative {
    pub text: String,
    pub confidence: Option<f32>,
    pub words: Vec<TranscriptionWord>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionChunk {
    /// Location of this sequential recognition result in the input audio.
    pub time: Option<TimeSpan>,
    /// Hypotheses ordered from most to least probable.
    pub alternatives: Vec<TranscriptionAlternative>,
    pub language: Option<String>,
    pub channel: Option<u16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionResult {
    pub results: Vec<TranscriptionChunk>,
    pub audio_duration: Duration,
}

impl TranscriptionResult {
    pub fn empty(audio_duration: Duration) -> Self {
        Self {
            results: Vec::new(),
            audio_duration,
        }
    }
}
