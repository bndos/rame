use std::sync::RwLockReadGuard;

use candle_core::{DType, Storage};
use ort::session::SessionInputValue;
use ort::value::{PrimitiveTensorElementType, TensorRef};

use crate::session::ort::mapper::input::{TensorOrtInput as OrtInput, TensorOrtInputDevice};
use crate::tensor::Tensor;

pub(in crate::session::ort) type TensorOrtInput<'a> = OrtInput<'a, CpuTensorOrtInputDevice>;

pub(in crate::session::ort) struct CpuTensorOrtInputDevice;

impl TensorOrtInputDevice for CpuTensorOrtInputDevice {
    fn tensor_storage_to_ort<'s>(
        name: &str,
        tensor: &Tensor,
        storage: &'s RwLockReadGuard<'_, Storage>,
    ) -> ort::Result<(String, SessionInputValue<'s>)> {
        match tensor.dtype() {
            DType::F32 => tensor_storage_to_ort::<f32>(name, tensor, storage),
            DType::I32 => tensor_storage_to_ort::<i32>(name, tensor, storage),
            DType::I64 => tensor_storage_to_ort::<i64>(name, tensor, storage),
            dtype => Err(unsupported_dtype(name, "CPU", dtype)),
        }
    }
}

fn tensor_storage_to_ort<'s, T>(
    name: &str,
    tensor: &Tensor,
    storage: &'s RwLockReadGuard<'_, Storage>,
) -> ort::Result<(String, SessionInputValue<'s>)>
where
    T: candle_core::WithDType + PrimitiveTensorElementType + std::fmt::Debug,
{
    let shape = tensor.dims().to_vec();
    let Storage::Cpu(storage) = &**storage else {
        return Err(ort::Error::new(format!(
            "expected CPU tensor for `{name}`, got {device:?}",
            device = tensor.device()
        )));
    };
    let data = storage
        .as_slice::<T>()
        .map_err(|err| ort::Error::new(err.to_string()))?;
    TensorRef::from_array_view((shape, data)).map(|tensor| (name.to_string(), tensor.into()))
}

fn unsupported_dtype(name: &str, device: &str, dtype: DType) -> ort::Error {
    ort::Error::new(format!(
        "unsupported {device} tensor dtype for `{name}`: {dtype:?}"
    ))
}
