use crate::RameResult;
use crate::tensor::TensorMap;

/// Batched model inputs produced by preprocessing, plus per-item decoding metadata.
#[derive(Debug)]
pub struct ProcessedBatch<C = (), I = TensorMap> {
    /// Number of source items represented by `inputs`.
    pub len: usize,

    /// Named tensors passed to the inference session.
    pub inputs: I,

    /// One preprocessing context per source item.
    pub contexts: Vec<C>,
}

/// Converts a source input into backend-ready tensors.
pub trait Processor {
    /// Raw input accepted by this processor.
    type Source<'a>;

    /// Metadata produced during preprocessing and needed during decoding.
    type Context;

    /// Backend-ready input storage produced by this processor.
    type Inputs;

    fn process_many<'a>(
        &self,
        sources: &'a [Self::Source<'a>],
    ) -> RameResult<ProcessedBatch<Self::Context, Self::Inputs>>;
}
