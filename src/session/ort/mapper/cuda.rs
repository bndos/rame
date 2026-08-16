use std::sync::RwLockReadGuard;

use candle_core::{DType, Storage};
use ort::session::SessionInputValue;

use crate::session::ort::mapper::input::{TensorOrtInput as OrtInput, TensorOrtInputDevice};
use crate::tensor::Tensor;

pub(in crate::session::ort) type TensorOrtInput<'a> = OrtInput<'a, CudaTensorOrtInputDevice>;

pub(in crate::session::ort) struct CudaTensorOrtInputDevice;

impl TensorOrtInputDevice for CudaTensorOrtInputDevice {
    fn tensor_storage_to_ort<'s>(
        name: &str,
        tensor: &Tensor,
        _storage: &'s RwLockReadGuard<'_, Storage>,
    ) -> ort::Result<(String, SessionInputValue<'s>)> {
        match tensor.dtype() {
            DType::F32 | DType::I32 | DType::I64 => Err(ort::Error::new(format!(
                "CUDA tensor input binding is not implemented for `{name}`",
            ))),
            dtype => Err(ort::Error::new(format!(
                "unsupported CUDA tensor dtype for `{name}`: {dtype:?}"
            ))),
        }
    }
}
