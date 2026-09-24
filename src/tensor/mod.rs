mod base;

pub use base::{TensorError, TensorMap, TensorResult, from_array, to_array};
pub(crate) use candle_core::Var;
pub use candle_core::{D, DType, Device, DeviceLocation, Tensor, WithDType as TensorElement};
