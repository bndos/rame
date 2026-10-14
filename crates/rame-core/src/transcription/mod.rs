mod error;
mod model;
mod options;
mod result;
mod speaker;
mod timestamp;

pub use error::TranscriptionError;
pub use model::{TranscriptionInput, TranscriptionModel};
pub use options::{AudioChannelMode, SpeakerDiarizationOptions, TranscriptionOptions};
pub use result::{
    TranscriptionAlternative, TranscriptionChunk, TranscriptionResult, TranscriptionWord,
};
pub use speaker::SpeakerId;
pub use timestamp::{TimeSpan, TimestampGranularity};
