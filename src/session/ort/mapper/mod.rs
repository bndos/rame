use std::sync::RwLockReadGuard;

use candle_core::Storage;
use ort::session::SessionInputValue;

use crate::tensor::Tensor;

mod cpu;
#[cfg(feature = "onnxruntime-cuda")]
mod cuda;
mod input;
mod output;

pub(in crate::session::ort) enum TensorOrtInput<'a> {
    Cpu(cpu::TensorOrtInput<'a>),
    #[cfg(feature = "onnxruntime-cuda")]
    Cuda(cuda::TensorOrtInput<'a>),
}

impl TensorOrtInput<'_> {
    pub(in crate::session::ort) fn as_session_input(
        &self,
    ) -> ort::Result<(String, SessionInputValue<'_>)> {
        match self {
            Self::Cpu(input) => input.as_session_input(),
            #[cfg(feature = "onnxruntime-cuda")]
            Self::Cuda(input) => input.as_session_input(),
        }
    }
}

impl Tensor {
    pub(super) fn ort_input<'a>(&'a self, name: &'a str) -> ort::Result<TensorOrtInput<'a>> {
        let (storage, layout) = self.storage_and_layout();
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            return Err(ort::Error::new(format!(
                "expected contiguous tensor with zero start offset for `{name}`"
            )));
        }

        match &*storage {
            Storage::Cpu(_) => cpu_ort_input(name, self, storage),
            Storage::Cuda(_) => cuda_ort_input(name, self, storage),
            Storage::Metal(_) => Err(ort::Error::new(format!(
                "Metal tensor input binding is not supported for `{name}`"
            ))),
        }
    }
}

fn cpu_ort_input<'a>(
    name: &'a str,
    tensor: &'a Tensor,
    storage: RwLockReadGuard<'a, Storage>,
) -> ort::Result<TensorOrtInput<'a>> {
    Ok(TensorOrtInput::Cpu(input::TensorOrtInput::new(
        name, tensor, storage,
    )))
}

#[cfg(feature = "onnxruntime-cuda")]
fn cuda_ort_input<'a>(
    name: &'a str,
    tensor: &'a Tensor,
    storage: RwLockReadGuard<'a, Storage>,
) -> ort::Result<TensorOrtInput<'a>> {
    Ok(TensorOrtInput::Cuda(input::TensorOrtInput::new(
        name, tensor, storage,
    )))
}

#[cfg(not(feature = "onnxruntime-cuda"))]
fn cuda_ort_input<'a>(
    name: &'a str,
    _tensor: &'a Tensor,
    _storage: RwLockReadGuard<'a, Storage>,
) -> ort::Result<TensorOrtInput<'a>> {
    Err(ort::Error::new(format!(
        "CUDA tensor input binding is not supported for `{name}`"
    )))
}
