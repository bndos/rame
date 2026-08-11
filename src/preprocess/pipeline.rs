use std::fmt;

use crate::RameResult;

/// Backend that owns preprocessing state creation and finalization.
pub trait PreprocessBackend: Sized {
    /// Raw input consumed by this preprocessing state.
    type Source<'a>;
    /// Mutable batch threaded through every preprocessing op.
    type Batch<'a>;
    /// Output produced after all preprocessing ops have run over a batch.
    type Output;
    /// Typed operation IR compiled by this backend.
    type Op: PreprocessOp<Self>;

    fn batch<'a>(&self, sources: &'a [Self::Source<'a>]) -> RameResult<Self::Batch<'a>>;

    fn finish(&self, batch: Self::Batch<'_>) -> RameResult<Self::Output>;

    fn compile(&self, ops: &mut Vec<Self::Op>);

    fn process_many<'a>(
        &self,
        sources: &'a [Self::Source<'a>],
        ops: &[Self::Op],
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
    fn apply<'a>(&self, batch: &mut B::Batch<'a>) -> RameResult<()>;
}

/// Ordered preprocessing operation list.
#[derive(Clone)]
pub struct PreprocessPipeline<B>
where
    B: PreprocessBackend,
{
    backend: B,
    ops: Vec<B::Op>,
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

    pub fn add_op(mut self, op: impl Into<B::Op>) -> Self {
        self.ops.push(op.into());
        self
    }

    pub fn ops_len(&self) -> usize {
        self.ops.len()
    }

    pub fn compile(mut self) -> Self {
        self.backend.compile(&mut self.ops);
        self
    }

    pub fn process_many<'a>(&self, sources: &'a [B::Source<'a>]) -> RameResult<B::Output> {
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
