use std::fmt;

use crate::RameResult;

/// Backend that owns preprocessing state creation, lowering, and finalization.
pub trait PreprocessBackend: Sized {
    /// Raw input consumed by this preprocessing state.
    type Source<'a>;
    /// Runtime data threaded through every preprocessing op.
    type Data<'a>;
    /// Output produced after all preprocessing ops have run over a batch.
    type Output;

    fn input<'a>(&self, sources: &'a [Self::Source<'a>]) -> RameResult<Self::Data<'a>>;

    fn finish(&self, data: Self::Data<'_>) -> RameResult<Self::Output>;

    fn compile(&self, ops: Vec<Box<dyn PreprocessOp<Self>>>) -> Vec<Box<dyn PreprocessOp<Self>>> {
        ops
    }
}

/// Executable preprocessing operation for a concrete backend.
pub trait PreprocessOp<B>: fmt::Debug + Send + Sync
where
    B: PreprocessBackend,
{
    fn forward<'a>(&self, data: B::Data<'a>) -> RameResult<B::Data<'a>>;
}

/// Editable preprocessing pipeline before backend lowering.
pub struct PreprocessPipelineBuilder<B>
where
    B: PreprocessBackend,
{
    backend: B,
    ops: Vec<Box<dyn PreprocessOp<B>>>,
}

impl<B> PreprocessPipelineBuilder<B>
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
        self.ops.push(Box::new(op));
        self
    }

    pub fn ops_len(&self) -> usize {
        self.ops.len()
    }

    pub fn compile(self) -> PreprocessPipeline<B> {
        let Self { backend, ops } = self;
        let ops = backend.compile(ops);
        PreprocessPipeline { backend, ops }
    }
}

impl<B> fmt::Debug for PreprocessPipelineBuilder<B>
where
    B: PreprocessBackend + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreprocessPipelineBuilder")
            .field("backend", &self.backend)
            .field("ops_len", &self.ops_len())
            .finish()
    }
}

/// Runnable preprocessing pipeline after backend lowering.
pub struct PreprocessPipeline<B>
where
    B: PreprocessBackend,
{
    backend: B,
    ops: Vec<Box<dyn PreprocessOp<B>>>,
}

impl<B> PreprocessPipeline<B>
where
    B: PreprocessBackend,
{
    pub fn ops_len(&self) -> usize {
        self.ops.len()
    }

    pub fn process_many<'a>(&self, sources: &'a [B::Source<'a>]) -> RameResult<B::Output> {
        let mut data = self.backend.input(sources)?;

        for op in &self.ops {
            data = op.forward(data)?;
        }

        self.backend.finish(data)
    }
}

impl<B> fmt::Debug for PreprocessPipeline<B>
where
    B: PreprocessBackend + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreprocessPipeline")
            .field("backend", &self.backend)
            .field("ops_len", &self.ops_len())
            .finish()
    }
}
