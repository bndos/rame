use std::time::Duration;

use crate::RameResult;

use super::TranscriptionError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeSpan {
    start: Duration,
    end: Duration,
}

impl TimeSpan {
    pub fn new(start: Duration, end: Duration) -> RameResult<Self> {
        if end < start {
            return Err(TranscriptionError::InvalidTimeSpan { start, end }.into());
        }

        Ok(Self { start, end })
    }

    pub fn start(self) -> Duration {
        self.start
    }

    pub fn end(self) -> Duration {
        self.end
    }

    pub fn duration(self) -> Duration {
        self.end - self.start
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimestampGranularity {
    /// Do not request timestamp metadata.
    #[default]
    None,
    /// Request timestamps for transcription segments.
    Segment,
    /// Request timestamps for individual words.
    Word,
    /// Request every timestamp level supported by the model.
    All,
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::RameError;

    use super::{TimeSpan, TranscriptionError};

    #[test]
    fn validates_time_spans_when_constructed() {
        let span = TimeSpan::new(Duration::from_secs(2), Duration::from_secs(5)).unwrap();

        assert_eq!(span.start(), Duration::from_secs(2));
        assert_eq!(span.end(), Duration::from_secs(5));
        assert_eq!(span.duration(), Duration::from_secs(3));
    }

    #[test]
    fn rejects_time_spans_that_end_before_they_start() {
        assert!(matches!(
            TimeSpan::new(Duration::from_secs(5), Duration::from_secs(2)).unwrap_err(),
            RameError::Transcription(TranscriptionError::InvalidTimeSpan { start, end })
                if start == Duration::from_secs(5) && end == Duration::from_secs(2)
        ));
    }
}
