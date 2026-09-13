use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParakeetConfig {
    pub sample_rate: u32,
    pub feature_stride: Duration,
    pub subsampling_factor: usize,
    pub max_tokens_per_step: usize,
}

impl Default for ParakeetConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            feature_stride: Duration::from_millis(10),
            subsampling_factor: 8,
            max_tokens_per_step: 10,
        }
    }
}

impl ParakeetConfig {
    pub fn encoder_frame_duration(self) -> Duration {
        self.feature_stride.mul_f64(self.subsampling_factor as f64)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::ParakeetConfig;

    #[test]
    fn derives_encoder_frame_duration_from_model_configuration() {
        assert_eq!(
            ParakeetConfig::default().encoder_frame_duration(),
            Duration::from_millis(80)
        );
    }
}
