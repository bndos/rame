use std::sync::RwLockReadGuard;

use candle_core::Storage;
use ort::session::SessionInputValue;
use ort::value::PrimitiveTensorElementType;

use crate::session::ort::mapper::input::{TensorOrtInput as OrtInput, TensorOrtInputDevice};
use crate::tensor::Tensor;

pub(in crate::session::ort) type TensorOrtInput<'a> = OrtInput<'a, CudaTensorOrtInputDevice>;

pub(in crate::session::ort) struct CudaTensorOrtInputDevice;

impl TensorOrtInputDevice for CudaTensorOrtInputDevice {
    fn tensor_storage_to_ort<'s, T>(
        name: &str,
        _tensor: &Tensor,
        _storage: &'s RwLockReadGuard<'_, Storage>,
    ) -> ort::Result<(String, SessionInputValue<'s>)>
    where
        T: candle_core::WithDType + PrimitiveTensorElementType + std::fmt::Debug,
    {
        Err(ort::Error::new(format!(
            "CUDA tensor input binding is not implemented for `{name}`",
        )))
    }
}
