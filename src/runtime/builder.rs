use crate::RameResult;
use crate::runtime::{InferencePipeline, ModelArchitecture, ModelArtifact};
use crate::session::SessionBackend;
use crate::sources::ResolveModelSource;

#[derive(Debug, Clone, Copy)]
pub struct Missing;

/// Collects source and artifact choices before loading a typed model.
#[derive(Debug, Clone)]
pub struct ModelBuilder<M, S = Missing, A = Missing> {
    architecture: M,
    source: S,
    artifact: A,
}

impl<M> ModelBuilder<M> {
    pub fn new(architecture: M) -> Self {
        Self {
            architecture,
            source: Missing,
            artifact: Missing,
        }
    }
}

impl<M, S, A> ModelBuilder<M, S, A> {
    pub fn source<NextSource>(self, source: NextSource) -> ModelBuilder<M, NextSource, A> {
        ModelBuilder {
            architecture: self.architecture,
            source,
            artifact: self.artifact,
        }
    }

    pub fn artifact<NextArtifact>(
        self,
        artifact: NextArtifact,
    ) -> ModelBuilder<M, S, NextArtifact> {
        ModelBuilder {
            architecture: self.architecture,
            source: self.source,
            artifact,
        }
    }
}

impl<M, S, A> ModelBuilder<M, S, A>
where
    M: ModelArchitecture,
    S: ResolveModelSource,
    A: ModelArtifact<Architecture = M>,
{
    pub fn build(
        self,
    ) -> RameResult<
        InferencePipeline<M, A::Processor, <A::Backend as SessionBackend>::Session, A::Decoder>,
    > {
        let source = self.source.resolve_model_source()?;
        let config = self.artifact.default_session_config();
        let session = self.artifact.load_session(&source, config)?;
        let processor = self.artifact.processor();
        let decoder = self.artifact.decoder();

        Ok(InferencePipeline::new(
            self.architecture,
            processor,
            session,
            decoder,
        ))
    }
}
