use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};

use candle_core::Device as CandleDevice;
use ndarray::ArrayD;
use thiserror::Error;

pub type TensorResult<T> = Result<T, TensorError>;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    #[error(transparent)]
    Shape(#[from] ndarray::ShapeError),
}

/// Compute device used by tensors, sessions, and preprocessing backends.
#[derive(Debug, Clone)]
pub struct Device(CandleDevice);

/// Tensor data passed between processors, sessions, and decoders.
#[derive(Debug, Clone)]
pub struct Tensor(candle_core::Tensor);

/// Named tensor collection used at model execution boundaries.
#[derive(Debug, Clone, Default)]
pub struct TensorMap(BTreeMap<String, Tensor>);

impl TensorMap {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu()
    }
}

impl Device {
    pub fn cpu() -> Self {
        Self(CandleDevice::Cpu)
    }

    pub fn cuda(device_id: usize) -> TensorResult<Self> {
        CandleDevice::new_cuda(device_id)
            .map(Self)
            .map_err(Into::into)
    }
}

impl From<CandleDevice> for Device {
    fn from(device: CandleDevice) -> Self {
        Self(device)
    }
}

impl Deref for Device {
    type Target = CandleDevice;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Tensor {
    pub fn from_candle(tensor: candle_core::Tensor) -> Self {
        Self(tensor)
    }

    pub fn from_vec<S, T>(data: Vec<T>, shape: S) -> TensorResult<Self>
    where
        S: Into<candle_core::Shape>,
        T: candle_core::WithDType,
    {
        Self::from_vec_on_device(data, shape, &Device::cpu())
    }

    pub fn from_vec_on_device<S, T>(data: Vec<T>, shape: S, device: &Device) -> TensorResult<Self>
    where
        S: Into<candle_core::Shape>,
        T: candle_core::WithDType,
    {
        candle_core::Tensor::from_vec(data, shape, device)
            .map(Self)
            .map_err(Into::into)
    }

    pub fn from_array<T>(array: ArrayD<T>) -> TensorResult<Self>
    where
        T: candle_core::WithDType + Clone,
    {
        let shape = array.shape().to_vec();
        let data = array.iter().cloned().collect::<Vec<_>>();
        Self::from_vec(data, shape)
    }

    pub fn to_array<T>(&self) -> TensorResult<ArrayD<T>>
    where
        T: candle_core::WithDType + Clone,
    {
        let shape = self.dims().to_vec();
        let data = self
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<T>())?;
        ArrayD::from_shape_vec(shape, data).map_err(Into::into)
    }
}

impl Deref for Tensor {
    type Target = candle_core::Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for TensorMap {
    type Target = BTreeMap<String, Tensor>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for TensorMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for TensorMap {
    type Item = (String, Tensor);
    type IntoIter = std::collections::btree_map::IntoIter<String, Tensor>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
