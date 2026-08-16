use candle_core::cuda_backend::cudarc::driver::CudaSlice;
use candle_core::cuda_backend::{CudaDType, CudaDevice};
use candle_core::{Shape, Storage};

use super::base::Tensor;

impl Tensor {
    pub(crate) fn from_cuda<T, S>(data: CudaSlice<T>, shape: S, device: CudaDevice) -> Self
    where
        T: CudaDType,
        S: Into<Shape>,
    {
        let storage = Storage::Cuda(T::wrap_cuda_slice(data, device));
        Tensor(candle_core::Tensor::from_storage(
            storage,
            shape,
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }
}
