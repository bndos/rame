use std::path::PathBuf;

use super::ParakeetMlxModel;
use super::pipeline;
use crate::RameResult;
use crate::models::parakeet::ParakeetTdtConfig;
use crate::runtime::ModelLoader;
use crate::sources::ResolvedModelSource;
use crate::tokenization::SentencePieceDecoder;

pub use crate::models::parakeet::ParakeetFeatureExtractorConfig as Preprocess;

#[derive(Debug, Clone, Default)]
pub struct Artifact {
    pub files: Files,
    pub preprocess: Preprocess,
    pub weights: WeightFormat,
    pub decoding: DecodingMode,
}

impl Artifact {
    pub fn files(mut self, files: Files) -> Self {
        self.files = files;
        self
    }

    pub fn preprocess(mut self, preprocess: Preprocess) -> Self {
        self.preprocess = preprocess;
        self
    }

    pub fn weights(mut self, weights: WeightFormat) -> Self {
        self.weights = weights;
        self
    }

    pub fn decoding(mut self, decoding: DecodingMode) -> Self {
        self.decoding = decoding;
        self
    }

    fn resolve_artifact(
        &self,
        source: &ResolvedModelSource,
    ) -> RameResult<(PathBuf, ParakeetTdtConfig, SentencePieceDecoder)> {
        let model_path = source.join_artifact_path(&self.files.model)?;
        let config_path = source.join_artifact_path(&self.files.config)?;
        let tokenizer_path = source.join_artifact_path(&self.files.tokenizer)?;
        let decoder = SentencePieceDecoder::from_file(tokenizer_path)?;
        let config = ParakeetTdtConfig::from_checkpoint(&config_path, &decoder)?;
        Ok((model_path, config, decoder))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DecodingMode {
    #[default]
    Standard,
    Mlx,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WeightFormat {
    #[default]
    Dense,
    QuantizedQ8,
}

#[derive(Debug, Clone)]
pub struct Files {
    pub model: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
}

impl Default for Files {
    fn default() -> Self {
        Self {
            model: PathBuf::from("model.safetensors"),
            config: PathBuf::from("config.json"),
            tokenizer: PathBuf::from("tokenizer.model"),
        }
    }
}

impl ModelLoader for Artifact {
    type Runner = ParakeetMlxModel;

    fn load_resolved(self, source: ResolvedModelSource) -> RameResult<Self::Runner> {
        let (model_path, config, decoder) = self.resolve_artifact(&source)?;
        pipeline::load(
            self.preprocess,
            self.weights,
            self.decoding,
            config,
            decoder,
            &model_path,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{Artifact, DecodingMode, Preprocess, WeightFormat};
    use crate::models::parakeet::ParakeetTdtConfig;

    #[test]
    fn defaults_to_original_dense_checkpoint() {
        let artifact = Artifact::default();
        assert_eq!(artifact.weights, WeightFormat::Dense);
        assert_eq!(artifact.decoding, DecodingMode::Standard);
        assert_eq!(artifact.files.model.to_str(), Some("model.safetensors"));
        assert_eq!(artifact.files.tokenizer.to_str(), Some("tokenizer.model"));
        assert_eq!(artifact.preprocess, Preprocess::default());
        assert_eq!(
            artifact.preprocess,
            ParakeetTdtConfig::v0_6b_v3().feature_extractor
        );
    }
}
