use std::marker::PhantomData;
use std::sync::RwLockReadGuard;

use candle_core::Storage;
use ort::session::SessionInputValue;

use crate::tensor::Tensor;

pub(in crate::session::ort) trait TensorOrtInputDevice {
    fn tensor_storage_to_ort<'s>(
        name: &str,
        tensor: &Tensor,
        storage: &'s RwLockReadGuard<'_, Storage>,
    ) -> ort::Result<(String, SessionInputValue<'s>)>;
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
        D::tensor_storage_to_ort(self.name, self.tensor, &self.storage)
    }
}
