use crate::RameResult;
use crate::tensor::TensorMap;

/// Model inputs produced by preprocessing, plus metadata needed by decoding.
#[derive(Debug, Clone)]
pub struct Processed<C = ()> {
    /// Named tensors passed to the inference session.
    pub inputs: TensorMap,

    /// Preprocessing metadata passed through to the decoder.
    pub context: C,
}

/// Converts a source input into backend-ready tensors.
pub trait Processor {
    /// Raw input accepted by this processor.
    type Source;

    /// Metadata produced during preprocessing and needed during decoding.
    type Context;

    fn process(&self, source: &Self::Source) -> RameResult<Processed<Self::Context>>;
}
