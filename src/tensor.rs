use std::collections::BTreeMap;

use ndarray::ArrayD;

/// Named tensor collection used at model execution boundaries.
pub type TensorMap = BTreeMap<String, TensorValue>;

/// Tensor data passed between processors, sessions, and decoders.
#[derive(Debug, Clone)]
pub enum TensorValue {
    F32(ArrayD<f32>),
}
