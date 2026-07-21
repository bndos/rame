use std::path::Path;

use crate::RameResult;
use crate::runtime::{Decoder, ModelArchitecture, Processor};
use crate::session::SessionBackend;
use crate::sources::ResolvedModelSource;

/// Export-specific recipe for assembling a model pipeline.
///
/// An artifact binds one exported package layout to its compatible architecture,
/// backend, processor, and decoder. It constructs the compatible runtime
/// pieces, but it does not own the loaded instances or run inference.
pub trait ModelArtifact {
    type Architecture: ModelArchitecture;
    type Backend: SessionBackend;
    type Processor: Processor;
    type Decoder: Decoder<
            Output = <Self::Architecture as ModelArchitecture>::Output,
            Context = <Self::Processor as Processor>::Context,
        >;

    /// Artifact-relative path to the executable model file.
    fn model_file(&self) -> &Path;

    fn session_config(&self) -> <Self::Backend as SessionBackend>::Config;
    fn processor(&self) -> Self::Processor;
    fn decoder(&self) -> Self::Decoder;

    fn load_session(
        &self,
        source: &ResolvedModelSource,
    ) -> RameResult<<Self::Backend as SessionBackend>::Session> {
        let model_path = source.join_artifact_path(self.model_file())?;

        Self::Backend::load(&model_path, self.session_config())
    }
}
