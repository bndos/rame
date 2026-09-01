use crate::RameResult;
use crate::runtime::{ModelArchitecture, ModelRunner};
use crate::sources::ResolvedModelSource;

/// Loads one exported implementation of a semantic model.
///
/// Simple exports normally return [`crate::runtime::StandardModelRunner`].
/// Composite or autoregressive exports can return a custom runner that owns all
/// required sessions, state, and control flow.
pub trait ModelLoader<M>
where
    M: ModelArchitecture,
{
    type Runner: ModelRunner<Architecture = M>;

    fn load(self, architecture: M, source: ResolvedModelSource) -> RameResult<Self::Runner>;
}
