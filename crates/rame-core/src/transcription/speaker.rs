/// Opaque speaker identifier assigned by diarization.
///
/// Its value identifies a speaker within a transcription result.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpeakerId(String);

impl SpeakerId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}
