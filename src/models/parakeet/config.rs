use std::time::Duration;

use crate::runtime::transducer::tdt::TdtDecodingConfig;
use crate::tokenization::TokenId;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParakeetFeatureExtractorConfig {
    pub sample_rate: u32,
    pub feature_size: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub preemphasis: f32,
    pub padding_value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParakeetTdtConfig {
    pub feature_extractor: ParakeetFeatureExtractorConfig,
    pub subsampling_factor: usize,
    pub max_symbols_per_step: usize,
    pub num_token_classes: usize,
    pub blank_token_id: TokenId,
    pub durations: [usize; 5],
}

impl From<ParakeetTdtConfig> for TdtDecodingConfig {
    fn from(config: ParakeetTdtConfig) -> Self {
        Self::new(
            config.num_token_classes,
            config.blank_token_id,
            config.durations,
            config.max_symbols_per_step,
        )
    }
}

impl ParakeetTdtConfig {
    pub const fn v0_6b_v3() -> Self {
        Self {
            feature_extractor: ParakeetFeatureExtractorConfig {
                sample_rate: 16_000,
                feature_size: 128,
                hop_length: 160,
                n_fft: 512,
                win_length: 400,
                preemphasis: 0.97,
                padding_value: 0.0,
            },
            subsampling_factor: 8,
            max_symbols_per_step: 10,
            num_token_classes: 8_193,
            blank_token_id: 8_192,
            durations: [0, 1, 2, 3, 4],
        }
    }

    pub fn feature_stride(self) -> Duration {
        Duration::from_secs_f64(
            self.feature_extractor.hop_length as f64 / self.feature_extractor.sample_rate as f64,
        )
    }

    pub fn encoder_frame_duration(self) -> Duration {
        self.feature_stride()
            .mul_f64(self.subsampling_factor as f64)
    }

    pub fn joint_logit_count(self) -> usize {
        self.num_token_classes + self.durations.len()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::ParakeetTdtConfig;

    #[test]
    fn derives_encoder_frame_duration_from_model_configuration() {
        assert_eq!(
            ParakeetTdtConfig::v0_6b_v3().feature_stride(),
            Duration::from_millis(10)
        );
        assert_eq!(
            ParakeetTdtConfig::v0_6b_v3().encoder_frame_duration(),
            Duration::from_millis(80)
        );
    }

    #[test]
    fn describes_the_parakeet_tdt_output_contract() {
        let config = ParakeetTdtConfig::v0_6b_v3();

        assert_eq!(config.feature_extractor.sample_rate, 16_000);
        assert_eq!(config.feature_extractor.feature_size, 128);
        assert_eq!(config.feature_extractor.hop_length, 160);
        assert_eq!(config.feature_extractor.n_fft, 512);
        assert_eq!(config.feature_extractor.win_length, 400);
        assert_eq!(config.feature_extractor.preemphasis, 0.97);
        assert_eq!(config.feature_extractor.padding_value, 0.0);
        assert_eq!(config.subsampling_factor, 8);
        assert_eq!(config.max_symbols_per_step, 10);
        assert_eq!(config.num_token_classes, 8_193);
        assert_eq!(config.blank_token_id, 8_192);
        assert_eq!(config.durations, [0, 1, 2, 3, 4]);
        assert_eq!(config.joint_logit_count(), 8_198);
    }
}
