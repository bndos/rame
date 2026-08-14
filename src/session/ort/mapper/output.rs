use ort::session::SessionOutputs;
use ort::value::{DynValue, TensorElementType};

use crate::session::ort::OrtError;
use crate::tensor::{Tensor, TensorMap};
use crate::{RameError, RameResult};

impl TryFrom<SessionOutputs<'_>> for TensorMap {
    type Error = RameError;

    fn try_from(outputs: SessionOutputs<'_>) -> Result<Self, Self::Error> {
        let mut tensors = Self::new();
        for (name, value) in outputs {
            tensors.insert(name.to_string(), output_to_tensor(name, value)?);
        }
        Ok(tensors)
    }
}

fn output_to_tensor(name: &str, value: DynValue) -> RameResult<Tensor> {
    match value.dtype().tensor_type() {
        Some(TensorElementType::Float32) => extract_output_tensor::<f32>(name, value),
        Some(TensorElementType::Int32) => extract_output_tensor::<i32>(name, value),
        Some(TensorElementType::Int64) => extract_output_tensor::<i64>(name, value),
        Some(tensor_type) => Err(OrtError::UnsupportedTensorType {
            name: name.to_string(),
            tensor_type: tensor_type.to_string(),
        }
        .into()),
        None => Err(OrtError::UnsupportedTensorType {
            name: name.to_string(),
            tensor_type: value.dtype().to_string(),
        }
        .into()),
    }
}

fn extract_output_tensor<T>(name: &str, value: DynValue) -> RameResult<Tensor>
where
    T: ort::value::PrimitiveTensorElementType + candle_core::WithDType + Clone,
{
    Tensor::from_array(
        value
            .try_extract_array::<T>()
            .map_err(OrtError::from)?
            .to_owned(),
    )
    .map_err(|err| {
        OrtError::InvalidInput {
            name: name.to_string(),
            reason: err.to_string(),
        }
        .into()
    })
}
