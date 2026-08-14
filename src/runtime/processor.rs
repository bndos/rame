use crate::RameResult;
use crate::tensor::TensorMap;

/// Batched model inputs produced by preprocessing, plus per-item decoding metadata.
#[derive(Debug)]
pub struct ProcessedBatch<C = ()> {
    /// Number of source items represented by `inputs`.
    pub len: usize,

    /// Named tensors passed to the inference session.
    pub inputs: TensorMap,

    /// One preprocessing context per source item.
    pub contexts: Vec<C>,
}

/// Converts a source input into backend-ready tensors.
pub trait Processor {
    /// Raw input accepted by this processor.
    type Source<'a>;

    /// Metadata produced during preprocessing and needed during decoding.
    type Context;

    fn process_many<'a>(
        &self,
        sources: &'a [Self::Source<'a>],
    ) -> RameResult<ProcessedBatch<Self::Context>>;
}
