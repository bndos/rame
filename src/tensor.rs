use std::ops::Deref;
use std::{collections::BTreeMap, ops::DerefMut};

use candle_core::Device;
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

impl Tensor {
    pub fn from_candle(tensor: candle_core::Tensor) -> Self {
        Self(tensor)
    }

    pub fn from_vec<S, T>(data: Vec<T>, shape: S) -> TensorResult<Self>
    where
        S: Into<candle_core::Shape>,
        T: candle_core::WithDType,
    {
        candle_core::Tensor::from_vec(data, shape, &Device::Cpu)
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
