use std::fmt;
use std::sync::Arc;

use crate::RameResult;

/// Backend that owns preprocessing state creation and finalization.
pub trait PreprocessBackend {
    /// Raw input consumed by this preprocessing state.
    type Source;
    /// Mutable state threaded through every preprocessing op.
    type State;
    /// Output produced after all preprocessing ops have run.
    type Output;

    fn state(&self, source: &Self::Source) -> RameResult<Self::State>;

    fn finish(&self, state: Self::State) -> RameResult<Self::Output>;
}

/// Executable preprocessing operation for a concrete backend.
pub trait PreprocessOp<B>: fmt::Debug + Send + Sync
where
    B: PreprocessBackend,
{
    fn apply(&self, state: &mut B::State) -> RameResult<()>;
}

/// Ordered preprocessing operation list.
#[derive(Clone)]
pub struct PreprocessPipeline<B>
where
    B: PreprocessBackend,
{
    backend: B,
    ops: Vec<Arc<dyn PreprocessOp<B>>>,
}

impl<B> PreprocessPipeline<B>
where
    B: PreprocessBackend,
{
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            ops: Vec::new(),
        }
    }

    pub fn add_op(mut self, op: impl PreprocessOp<B> + 'static) -> Self {
        self.ops.push(Arc::new(op));
        self
    }

    pub fn ops_len(&self) -> usize {
        self.ops.len()
    }

    pub fn process(&self, source: &B::Source) -> RameResult<B::Output> {
        let mut state = self.backend.state(source)?;
        for op in &self.ops {
            op.apply(&mut state)?;
        }
        self.backend.finish(state)
    }
}

impl<B> fmt::Debug for PreprocessPipeline<B>
where
    B: PreprocessBackend + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreprocessPipeline")
            .field("backend", &self.backend)
            .field("ops_len", &self.ops.len())
            .finish()
    }
}
