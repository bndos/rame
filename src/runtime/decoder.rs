use crate::RameResult;
use crate::tensor::TensorMap;

/// Converts backend output tensors into a typed model result.
pub trait Decoder {
    /// Typed result produced by this decoder.
    type Output;

    /// Metadata produced during preprocessing and needed during decoding.
    type Context;

    fn decode(&self, outputs: &TensorMap, context: &Self::Context) -> RameResult<Self::Output>;
}
