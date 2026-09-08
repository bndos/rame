use crate::RameResult;
use crate::runtime::ModelRunner;
use crate::sources::{ResolveModelSource, ResolvedModelSource};

/// Loads one exported model implementation.
///
/// Simple exports normally return [`crate::runtime::StandardModelRunner`].
/// Composite or autoregressive exports can return a custom runner that owns all
/// required sessions, state, and control flow.
pub trait ModelLoader {
    type Runner: ModelRunner;

    fn load(self, source: impl ResolveModelSource) -> RameResult<Self::Runner>
    where
        Self: Sized,
    {
        self.load_resolved(source.resolve_model_source()?)
    }

    fn load_resolved(self, source: ResolvedModelSource) -> RameResult<Self::Runner>;
}
