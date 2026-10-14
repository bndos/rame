use std::num::{NonZeroU16, NonZeroU32};
use std::time::Duration;

use thiserror::Error;

use crate::RameResult;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AudioError {
    #[error("audio sample rate must be greater than zero")]
    InvalidSampleRate,

    #[error("audio channel count must be greater than zero")]
    InvalidChannelCount,

    #[error(
        "interleaved audio has {samples} samples, which is not divisible by {channels} channels"
    )]
    IncompleteFrame { samples: usize, channels: u16 },

    #[error("audio frame range {start}..{end} is invalid for {frames} frames")]
    InvalidFrameRange {
        start: usize,
        end: usize,
        frames: usize,
    },
}

/// Interleaved floating-point PCM backed by caller-selected storage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioBuffer<D> {
    data: D,
    sample_rate: NonZeroU32,
    channels: NonZeroU16,
}

/// Owned interleaved floating-point PCM.
pub type Audio = AudioBuffer<Vec<f32>>;

/// Borrowed interleaved floating-point PCM.
pub type AudioView<'a> = AudioBuffer<&'a [f32]>;

impl<D> AudioBuffer<D> {
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate.get()
    }

    pub fn channels(&self) -> u16 {
        self.channels.get()
    }

    pub fn into_data(self) -> D {
        self.data
    }
}

impl<D> AudioBuffer<D>
where
    D: AsRef<[f32]>,
{
    pub fn from_interleaved_f32(sample_rate: u32, channels: u16, data: D) -> RameResult<Self> {
        let (sample_rate, channels) = validate(sample_rate, channels, data.as_ref().len())?;
        Ok(Self {
            data,
            sample_rate,
            channels,
        })
    }

    pub fn data(&self) -> &[f32] {
        self.data.as_ref()
    }

    pub fn frames(&self) -> usize {
        self.data().len() / usize::from(self.channels.get())
    }

    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(self.frames() as f64 / f64::from(self.sample_rate.get()))
    }

    /// Returns a frame-aligned subview without copying PCM data.
    pub fn slice_frames(&self, start: usize, end: usize) -> RameResult<AudioView<'_>> {
        let frames = self.frames();
        if start > end || end > frames {
            return Err(AudioError::InvalidFrameRange { start, end, frames }.into());
        }
        let channels = usize::from(self.channels.get());
        AudioView::from_interleaved_f32(
            self.sample_rate.get(),
            self.channels.get(),
            &self.data()[start * channels..end * channels],
        )
    }
}

impl Audio {
    pub fn as_view(&self) -> AudioView<'_> {
        AudioView {
            data: &self.data,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }
}

impl<'a> AudioView<'a> {
    pub fn to_owned_audio(&self) -> RameResult<Audio> {
        Audio::from_interleaved_f32(self.sample_rate(), self.channels(), self.data().to_vec())
    }
}

fn validate(
    sample_rate: u32,
    channels: u16,
    samples: usize,
) -> RameResult<(NonZeroU32, NonZeroU16)> {
    let sample_rate = NonZeroU32::new(sample_rate).ok_or(AudioError::InvalidSampleRate)?;
    let channels = NonZeroU16::new(channels).ok_or(AudioError::InvalidChannelCount)?;
    if !samples.is_multiple_of(usize::from(channels.get())) {
        return Err(AudioError::IncompleteFrame {
            samples,
            channels: channels.get(),
        }
        .into());
    }
    Ok((sample_rate, channels))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::RameError;

    use super::{AudioError, AudioView};

    #[test]
    fn validates_and_slices_interleaved_audio_without_copying() {
        let samples = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioView::from_interleaved_f32(2, 2, &samples).unwrap();

        assert_eq!(audio.frames(), 3);
        assert_eq!(audio.duration(), Duration::from_secs_f64(1.5));

        let tail = audio.slice_frames(1, 3).unwrap();
        assert_eq!(tail.data(), &samples[2..]);
        assert_eq!(tail.data().as_ptr(), samples[2..].as_ptr());
    }

    #[test]
    fn rejects_partial_interleaved_frames() {
        assert!(matches!(
            AudioView::from_interleaved_f32(16_000, 2, &[0.0, 1.0, 2.0]).unwrap_err(),
            RameError::Audio(AudioError::IncompleteFrame {
                samples: 3,
                channels: 2,
            })
        ));
    }

    #[test]
    fn owned_and_borrowed_audio_share_the_same_contract() {
        let audio = super::Audio::from_interleaved_f32(16_000, 1, vec![0.0; 16_000]).unwrap();

        assert_eq!(audio.frames(), 16_000);
        assert_eq!(audio.duration(), Duration::from_secs(1));
        assert_eq!(audio.as_view().data().as_ptr(), audio.data().as_ptr());
    }
}
