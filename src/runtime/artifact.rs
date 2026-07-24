use std::path::PathBuf;

use crate::runtime::{Decoder, ModelArchitecture, Processor};
use crate::session::SessionBackend;

/// Owned runtime pieces produced by consuming an artifact recipe.
#[derive(Debug, Clone)]
pub struct ArtifactParts<C, P, D> {
    pub model_file: PathBuf,
    pub session_config: C,
    pub processor: P,
    pub decoder: D,
}

/// Export-specific recipe for assembling a model pipeline.
///
/// An artifact binds one exported package layout to its compatible architecture,
/// backend, processor, and decoder. Consuming it moves its config into owned
/// runtime pieces.
pub trait ModelArtifact {
    type Architecture: ModelArchitecture;
    type Backend: SessionBackend;
    type Processor: Processor;
    type Decoder: Decoder<
            Output = <Self::Architecture as ModelArchitecture>::Output,
            Context = <Self::Processor as Processor>::Context,
        >;

    fn into_parts(
        self,
    ) -> ArtifactParts<<Self::Backend as SessionBackend>::Config, Self::Processor, Self::Decoder>;
}
