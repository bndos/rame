use crate::RameError;
use crate::RameResult;
use crate::tensor::TensorMap;

/// Batched inference outputs plus per-item preprocessing metadata.
pub struct DecodeBatch<'a, C> {
    pub len: usize,
    pub outputs: &'a TensorMap,
    pub contexts: &'a [C],
}

/// Converts backend output tensors into a typed model result.
pub trait Decoder {
    /// Typed result produced by this decoder.
    type Output;

    /// Metadata produced during preprocessing and needed during decoding.
    type Context;

    fn decode_batch(&self, batch: DecodeBatch<'_, Self::Context>) -> RameResult<Vec<Self::Output>>;

    fn decode(&self, outputs: &TensorMap, context: &Self::Context) -> RameResult<Self::Output> {
        let decoded = self.decode_batch(DecodeBatch {
            len: 1,
            outputs,
            contexts: std::slice::from_ref(context),
        })?;
        let [output]: [_; 1] = decoded.try_into().map_err(|outputs: Vec<Self::Output>| {
            RameError::InvalidBatchLength {
                stage: "decoder output",
                expected: 1,
                actual: outputs.len(),
            }
        })?;

        Ok(output)
    }
}
