use std::path::Path;
use std::time::Duration;

use serde::Deserialize;

use crate::runtime::transducer::tdt::TdtDecodingConfig;
use crate::tokenization::TokenId;
use crate::{RameResult, models::ModelError, tokenization::SentencePieceDecoder};

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

impl Default for ParakeetFeatureExtractorConfig {
    fn default() -> Self {
        Self::v0_6b_v3()
    }
}

impl ParakeetFeatureExtractorConfig {
    pub const fn v0_6b_v3() -> Self {
        Self {
            sample_rate: 16_000,
            feature_size: 128,
            hop_length: 160,
            n_fft: 512,
            win_length: 400,
            preemphasis: 0.97,
            padding_value: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParakeetTdtConfig {
    pub feature_extractor: ParakeetFeatureExtractorConfig,
    pub network: ParakeetNetworkConfig,
    pub subsampling_factor: usize,
    pub max_symbols_per_step: usize,
    pub num_token_classes: usize,
    pub blank_token_id: TokenId,
    pub durations: [usize; 5],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParakeetNetworkConfig {
    pub prediction_hidden: usize,
    pub prediction_layers: usize,
    pub joint_hidden: usize,
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
            feature_extractor: ParakeetFeatureExtractorConfig::v0_6b_v3(),
            network: ParakeetNetworkConfig {
                prediction_hidden: 640,
                prediction_layers: 2,
                joint_hidden: 640,
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

#[derive(Deserialize)]
struct CheckpointConfig {
    encoder: CheckpointEncoder,
    joint: CheckpointJoint,
    decoder: CheckpointPredictor,
    decoding: CheckpointDecoding,
}

#[derive(Deserialize)]
struct CheckpointEncoder {
    d_model: usize,
}

#[derive(Deserialize)]
struct CheckpointJoint {
    vocabulary: Vec<String>,
    num_classes: usize,
    num_extra_outputs: usize,
    jointnet: CheckpointJointNetwork,
}

#[derive(Deserialize)]
struct CheckpointJointNetwork {
    joint_hidden: usize,
    encoder_hidden: usize,
    pred_hidden: usize,
}

#[derive(Deserialize)]
struct CheckpointPredictor {
    vocab_size: usize,
    prednet: CheckpointPredictionNetwork,
}

#[derive(Deserialize)]
struct CheckpointPredictionNetwork {
    pred_hidden: usize,
    pred_rnn_layers: usize,
}

#[derive(Deserialize)]
struct CheckpointDecoding {
    model_type: String,
    durations: Vec<usize>,
    greedy: CheckpointGreedy,
}

#[derive(Deserialize)]
struct CheckpointGreedy {
    max_symbols: usize,
}

impl CheckpointConfig {
    fn network_config(&self) -> ParakeetNetworkConfig {
        ParakeetNetworkConfig {
            prediction_hidden: self.decoder.prednet.pred_hidden,
            prediction_layers: self.decoder.prednet.pred_rnn_layers,
            joint_hidden: self.joint.jointnet.joint_hidden,
        }
    }
}

impl ParakeetTdtConfig {
    pub fn from_checkpoint(path: &Path, tokenizer: &SentencePieceDecoder) -> RameResult<Self> {
        let content = std::fs::read(path).map_err(|error| {
            ModelError::InvalidArtifact(format!(
                "failed to read Parakeet config {}: {error}",
                path.display()
            ))
        })?;
        let checkpoint: CheckpointConfig = serde_json::from_slice(&content).map_err(|error| {
            ModelError::InvalidArtifact(format!(
                "failed to parse Parakeet config {}: {error}",
                path.display()
            ))
        })?;
        let mut config = Self::v0_6b_v3();
        config.network = checkpoint.network_config();
        config.validate_checkpoint(&checkpoint, tokenizer)?;
        Ok(config)
    }

    fn validate_checkpoint(
        &self,
        checkpoint: &CheckpointConfig,
        tokenizer: &SentencePieceDecoder,
    ) -> RameResult<()> {
        let count = checkpoint.joint.vocabulary.len();
        let blank = u32::try_from(count)
            .map_err(|_| ModelError::InvalidArtifact("Parakeet vocabulary is too large".into()))?;
        if checkpoint.joint.num_classes != count
            || checkpoint.decoder.vocab_size != count
            || checkpoint.joint.num_extra_outputs != self.durations.len()
            || checkpoint.decoding.model_type != "tdt"
            || checkpoint.decoding.durations.as_slice() != self.durations
            || checkpoint.decoding.greedy.max_symbols != self.max_symbols_per_step
            || self.blank_token_id != blank
            || count.checked_add(1) != Some(self.num_token_classes)
            || checkpoint.encoder.d_model != 1024
            || checkpoint.joint.jointnet.encoder_hidden != checkpoint.encoder.d_model
            || self.network.prediction_hidden == 0
            || self.network.prediction_layers == 0
            || self.network.prediction_layers > 32
            || self.network.joint_hidden == 0
            || self
                .network
                .prediction_hidden
                .checked_mul(4)
                .and_then(|value| i32::try_from(value).ok())
                .is_none()
            || i32::try_from(self.network.joint_hidden).is_err()
            || i32::try_from(self.num_token_classes).is_err()
            || i32::try_from(self.joint_logit_count()).is_err()
            || checkpoint.joint.jointnet.pred_hidden != self.network.prediction_hidden
            || checkpoint.joint.jointnet.joint_hidden != self.network.joint_hidden
        {
            return Err(ModelError::InvalidArtifact(
                "Parakeet checkpoint does not match the 0.6B v3 TDT configuration".into(),
            )
            .into());
        }
        if count != tokenizer.vocabulary_size() {
            return Err(ModelError::InvalidArtifact(format!(
                "Parakeet vocabulary has {count} entries, tokenizer has {}",
                tokenizer.vocabulary_size()
            ))
            .into());
        }
        for (index, piece) in checkpoint.joint.vocabulary.iter().enumerate() {
            let actual = if piece == "<unk>" {
                Some(tokenizer.unknown_id())
            } else {
                tokenizer.piece_id(piece)?
            };
            if actual != u32::try_from(index).ok() {
                return Err(ModelError::InvalidArtifact(format!(
                    "Parakeet vocabulary token {index} does not match tokenizer.model"
                ))
                .into());
            }
        }
        Ok(())
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
        assert_eq!(config.network.prediction_hidden, 640);
        assert_eq!(config.network.prediction_layers, 2);
        assert_eq!(config.network.joint_hidden, 640);
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::CheckpointConfig;

    #[test]
    fn parses_checkpoint_metadata() {
        let config: CheckpointConfig = serde_json::from_str(
            r#"{"encoder":{"d_model":1024},"joint":{"vocabulary":["<unk>","a"],"num_classes":2,"num_extra_outputs":5,"jointnet":{"joint_hidden":384,"encoder_hidden":1024,"pred_hidden":256}},"decoder":{"vocab_size":2,"prednet":{"pred_hidden":256,"pred_rnn_layers":3}},"decoding":{"model_type":"tdt","durations":[0,1,2,3,4],"greedy":{"max_symbols":10}}}"#,
        )
        .unwrap();
        assert_eq!(config.joint.vocabulary, ["<unk>", "a"]);
        assert_eq!(config.joint.num_classes, 2);
        assert_eq!(config.decoder.vocab_size, 2);
        assert_eq!(config.network_config().prediction_hidden, 256);
        assert_eq!(config.network_config().prediction_layers, 3);
        assert_eq!(config.network_config().joint_hidden, 384);
        assert_eq!(config.decoding.durations, [0, 1, 2, 3, 4]);
    }
}
