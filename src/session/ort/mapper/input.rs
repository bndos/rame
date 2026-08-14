use std::marker::PhantomData;
use std::sync::RwLockReadGuard;

use candle_core::{DType, Storage};
use ort::session::SessionInputValue;
use ort::value::PrimitiveTensorElementType;

use crate::tensor::Tensor;

pub(in crate::session::ort) trait TensorOrtInputDevice {
    fn tensor_storage_to_ort<'s, T>(
        name: &str,
        tensor: &Tensor,
        storage: &'s RwLockReadGuard<'_, Storage>,
    ) -> ort::Result<(String, SessionInputValue<'s>)>
    where
        T: candle_core::WithDType + PrimitiveTensorElementType + std::fmt::Debug;
}

pub(in crate::session::ort) struct TensorOrtInput<'a, D> {
    name: &'a str,
    tensor: &'a Tensor,
    storage: RwLockReadGuard<'a, Storage>,
    device: PhantomData<D>,
}

impl<'a, D> TensorOrtInput<'a, D> {
    pub(super) fn new(
        name: &'a str,
        tensor: &'a Tensor,
        storage: RwLockReadGuard<'a, Storage>,
    ) -> Self {
        Self {
            name,
            tensor,
            storage,
            device: PhantomData,
        }
    }
}

impl<D> TensorOrtInput<'_, D>
where
    D: TensorOrtInputDevice,
{
    pub(super) fn as_session_input(&self) -> ort::Result<(String, SessionInputValue<'_>)> {
        match self.tensor.dtype() {
            DType::F32 => D::tensor_storage_to_ort::<f32>(self.name, self.tensor, &self.storage),
            DType::I32 => D::tensor_storage_to_ort::<i32>(self.name, self.tensor, &self.storage),
            DType::I64 => D::tensor_storage_to_ort::<i64>(self.name, self.tensor, &self.storage),
            dtype => Err(ort::Error::new(format!(
                "unsupported tensor dtype for `{name}`: {dtype:?}",
                name = self.name
            ))),
        }
    }
}
