use crate::tensor::DType;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreEmphasis {
    pub coefficient: f32,
}

impl PreEmphasis {
    pub fn new(coefficient: f32) -> Self {
        Self { coefficient }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogMelSpectrogram {
    pub sample_rate: u32,
    pub mel_bins: usize,
    pub fft_size: usize,
    pub window_size: usize,
    pub hop_length: usize,
    pub log_guard: f32,
}

impl LogMelSpectrogram {
    pub const DEFAULT_LOG_GUARD: f32 = 1.0e-5;

    pub fn new(
        sample_rate: u32,
        mel_bins: usize,
        fft_size: usize,
        window_size: usize,
        hop_length: usize,
    ) -> Self {
        Self {
            sample_rate,
            mel_bins,
            fft_size,
            window_size,
            hop_length,
            log_guard: Self::DEFAULT_LOG_GUARD,
        }
    }

    pub(crate) fn hann_window(self) -> Vec<f32> {
        const HANN_OFFSET: f32 = 0.5;
        const HANN_AMPLITUDE: f32 = 0.5;
        const FULL_CIRCLE_RADIANS: f32 = 2.0 * std::f32::consts::PI;

        let mut window = (0..self.window_size)
            .map(|index| {
                HANN_OFFSET
                    - HANN_AMPLITUDE
                        * (FULL_CIRCLE_RADIANS * index as f32 / self.window_size as f32).cos()
            })
            .collect::<Vec<_>>();
        window.resize(self.fft_size, 0.0);
        window
    }

    pub(crate) fn slaney_filterbank(self) -> Vec<f32> {
        const SLANEY_TRIANGLE_AREA_SCALE: f32 = 2.0;

        let frequencies = (0..=self.fft_size / 2)
            .map(|index| index as f32 * self.sample_rate as f32 / self.fft_size as f32)
            .collect::<Vec<_>>();
        let min_mel = hz_to_slaney_mel(0.0);
        let max_mel = hz_to_slaney_mel(self.sample_rate as f32 / 2.0);
        let mel_frequencies = (0..self.mel_bins + 2)
            .map(|index| {
                let mel = min_mel + (max_mel - min_mel) * index as f32 / (self.mel_bins + 1) as f32;
                slaney_mel_to_hz(mel)
            })
            .collect::<Vec<_>>();
        let mut output = vec![0.0; self.mel_bins * frequencies.len()];
        for mel in 0..self.mel_bins {
            let left = mel_frequencies[mel];
            let center = mel_frequencies[mel + 1];
            let right = mel_frequencies[mel + 2];
            let normalization = SLANEY_TRIANGLE_AREA_SCALE / (right - left);
            for (bin, frequency) in frequencies.iter().copied().enumerate() {
                let lower = (frequency - left) / (center - left);
                let upper = (right - frequency) / (right - center);
                output[mel * frequencies.len() + bin] = lower.min(upper).max(0.0) * normalization;
            }
        }
        output
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizeFeatures {
    pub epsilon: f32,
    pub output_dtype: DType,
}

impl NormalizeFeatures {
    pub const DEFAULT_EPSILON: f32 = 1.0e-5;

    pub fn new(output_dtype: DType) -> Self {
        Self {
            epsilon: Self::DEFAULT_EPSILON,
            output_dtype,
        }
    }
}

fn hz_to_slaney_mel(frequency: f32) -> f32 {
    let linear_step = 200.0 / 3.0;
    if frequency < 1_000.0 {
        frequency / linear_step
    } else {
        15.0 + (frequency / 1_000.0).ln() / (6.4f32.ln() / 27.0)
    }
}

fn slaney_mel_to_hz(mel: f32) -> f32 {
    let linear_step = 200.0 / 3.0;
    if mel < 15.0 {
        mel * linear_step
    } else {
        1_000.0 * ((mel - 15.0) * (6.4f32.ln() / 27.0)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::LogMelSpectrogram;

    #[test]
    fn builds_configured_window_and_filterbank() {
        let op = LogMelSpectrogram::new(16_000, 128, 512, 400, 160);
        assert_eq!(op.hann_window().len(), 512);
        assert_eq!(op.slaney_filterbank().len(), 128 * 257);
    }
}
