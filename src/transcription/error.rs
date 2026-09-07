use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TranscriptionError {
    #[error("transcription time span end {end:?} precedes start {start:?}")]
    InvalidTimeSpan { start: Duration, end: Duration },

    #[error("transcription option `{option}` is not supported by this model")]
    UnsupportedOption { option: &'static str },
}
