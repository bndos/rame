use std::fmt;
use std::sync::Arc;

use crate::RameResult;

/// Backend that owns preprocessing state creation and finalization.
pub trait PreprocessBackend: Sized {
    /// Raw input consumed by this preprocessing state.
    type Source;
    /// Mutable batch threaded through every preprocessing op.
    type Batch;
    /// Output produced after all preprocessing ops have run over a batch.
    type Output;

    fn batch(&self, sources: &[Self::Source]) -> RameResult<Self::Batch>;

    fn finish(&self, batch: Self::Batch) -> RameResult<Self::Output>;

    fn compile(&self, ops: &mut Vec<Arc<dyn PreprocessOp<Self>>>);

    fn process_many(
        &self,
        sources: &[Self::Source],
        ops: &[Arc<dyn PreprocessOp<Self>>],
    ) -> RameResult<Self::Output> {
        let mut batch = self.batch(sources)?;

        for op in ops {
            op.apply(&mut batch)?;
        }

        self.finish(batch)
    }
}

/// Executable preprocessing operation for a concrete backend.
pub trait PreprocessOp<B>: fmt::Debug + Send + Sync
where
    B: PreprocessBackend,
{
    fn apply(&self, batch: &mut B::Batch) -> RameResult<()>;
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

    pub fn compile(mut self) -> Self {
        self.backend.compile(&mut self.ops);
        self
    }

    pub fn process_many(&self, sources: &[B::Source]) -> RameResult<B::Output> {
        self.backend.process_many(sources, &self.ops)
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
