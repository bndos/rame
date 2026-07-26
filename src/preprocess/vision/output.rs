use ndarray::{Array2, Array4};

/// Batched image tensor and per-image metadata produced by vision preprocessing.
#[derive(Debug, Clone)]
pub struct VisionBatchOutput {
    pub tensor: Array4<f32>,
    pub image_shapes: Array2<f32>,
    pub scale_factors: Array2<f32>,
}

impl VisionBatchOutput {
    pub fn len(&self) -> usize {
        self.tensor.shape()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
