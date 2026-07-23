use ndarray::Array4;

/// Tensor and image metadata produced by vision preprocessing.
#[derive(Debug, Clone)]
pub struct VisionTensorOutput {
    pub tensor: Array4<f32>,
    pub scale_factor: [f32; 2],
}
