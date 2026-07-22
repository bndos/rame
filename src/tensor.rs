use std::collections::BTreeMap;

use ndarray::ArrayD;

/// Named tensor collection used at model execution boundaries.
pub type TensorMap = BTreeMap<String, TensorValue>;

/// Tensor data passed between processors, sessions, and decoders.
#[derive(Debug, Clone)]
pub enum TensorValue {
    F32(ArrayD<f32>),
    I64(ArrayD<i64>),
}

impl TensorValue {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::F32(_) => "f32",
            Self::I64(_) => "i64",
        }
    }
}
