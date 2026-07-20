use hf_hub::{HFClientSync, split_id};

use super::{ResolveModelSource, ResolvedModelSource, SourceError};
use crate::RameResult;

/// Configured Hugging Face provider used to create model sources.
#[derive(Debug, Clone)]
pub struct HuggingFace {
    /// `HFClientSync` is cloneable and already shares its HTTP client internally.
    client: HFClientSync,
}

impl HuggingFace {
    pub fn new() -> RameResult<Self> {
        HFClientSync::new()
            .map(Self::from_client)
            .map_err(|err| SourceError::HuggingFace(err.to_string()).into())
    }

    pub fn from_client(client: HFClientSync) -> Self {
        Self { client }
    }

    pub fn model(&self, repo: impl Into<String>) -> HuggingFaceModel {
        HuggingFaceModel {
            client: self.client.clone(),
            repo: repo.into(),
            revision: None,
        }
    }
}

/// Source for one model repository on a Hugging Face provider.
#[derive(Debug, Clone)]
pub struct HuggingFaceModel {
    client: HFClientSync,
    repo: String,
    revision: Option<String>,
}

impl HuggingFaceModel {
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    pub fn repo(&self) -> &str {
        &self.repo
    }
}

impl ResolveModelSource for HuggingFaceModel {
    fn resolve_model_source(&self) -> RameResult<ResolvedModelSource> {
        let (owner, name) = split_id(self.repo());
        let repo = self.client.model(owner, name);

        repo.snapshot_download()
            .maybe_revision(self.revision.clone())
            .send()
            .map(ResolvedModelSource::new)
            .map_err(|err| SourceError::HuggingFace(err.to_string()).into())
    }
}
