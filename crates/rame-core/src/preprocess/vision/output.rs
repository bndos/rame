use crate::tensor::Tensor;

/// Batched image tensor and per-image metadata produced by vision preprocessing.
#[derive(Debug)]
pub struct VisionBatchOutput {
    pub tensor: Tensor,
    pub image_shapes: Tensor,
    pub scale_factors: Tensor,
}

impl VisionBatchOutput {
    pub fn len(&self) -> usize {
        self.tensor.dims()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
