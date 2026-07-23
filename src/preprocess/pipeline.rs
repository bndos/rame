use std::sync::Arc;
use std::{fmt, marker::PhantomData};

use crate::RameResult;

/// Temporary mutable state used while a preprocessing pipeline runs.
pub trait PreprocessState: Sized {
    /// Raw input consumed by this preprocessing state.
    type Source;

    fn new(source: &Self::Source) -> RameResult<Self>;
}

/// Operation that mutates preprocessing state.
pub trait PreprocessOp<S>: fmt::Debug + Send + Sync {
    fn apply(&self, state: &mut S) -> RameResult<()>;
}

/// Converts completed preprocessing state into a typed output.
pub trait PreprocessFinalizer<S>: fmt::Debug + Send + Sync {
    /// Output produced after all preprocessing ops have run.
    type Output;

    fn finish(&self, state: S) -> RameResult<Self::Output>;
}

/// Ordered preprocessing pipeline for a concrete state and finalizer.
pub struct PreprocessPipeline<S, F> {
    ops: Vec<Arc<dyn PreprocessOp<S>>>,
    finalizer: F,
    state: PhantomData<S>,
}

impl<S, F> Clone for PreprocessPipeline<S, F>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            ops: self.ops.clone(),
            finalizer: self.finalizer.clone(),
            state: PhantomData,
        }
    }
}

impl<S, F> fmt::Debug for PreprocessPipeline<S, F>
where
    F: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreprocessPipeline")
            .field("ops_len", &self.ops.len())
            .field("finalizer", &self.finalizer)
            .finish()
    }
}

impl<S, F> PreprocessPipeline<S, F> {
    pub fn new(finalizer: F) -> Self {
        Self {
            ops: Vec::new(),
            finalizer,
            state: PhantomData,
        }
    }

    pub fn add_op(mut self, op: impl PreprocessOp<S> + 'static) -> Self {
        self.ops.push(Arc::new(op));
        self
    }
}

impl<S, F> PreprocessPipeline<S, F>
where
    S: PreprocessState,
    F: PreprocessFinalizer<S>,
{
    pub fn process(&self, source: &S::Source) -> RameResult<F::Output> {
        let mut state = S::new(source)?;

        for op in &self.ops {
            op.apply(&mut state)?;
        }

        self.finalizer.finish(state)
    }
}
