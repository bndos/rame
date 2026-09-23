use std::collections::BTreeMap;
use std::ops::Index;

use candle_core::{Tensor, WithDType};
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

/// Named tensor collection used at model execution boundaries.
#[derive(Debug, Clone, Default)]
pub struct TensorMap(BTreeMap<String, Tensor>);

impl TensorMap {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub fn insert(&mut self, name: String, tensor: Tensor) -> Option<Tensor> {
        self.0.insert(name, tensor)
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.0.get(name)
    }

    pub fn remove(&mut self, name: &str) -> Option<Tensor> {
        self.0.remove(name)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Index<&str> for TensorMap {
    type Output = Tensor;

    fn index(&self, name: &str) -> &Self::Output {
        &self.0[name]
    }
}

impl IntoIterator for TensorMap {
    type Item = (String, Tensor);
    type IntoIter = std::collections::btree_map::IntoIter<String, Tensor>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Copies an owned ndarray into a CPU Candle tensor.
pub fn from_array<T>(array: ArrayD<T>) -> TensorResult<Tensor>
where
    T: WithDType + Clone,
{
    let shape = array.shape().to_vec();
    let data = array.iter().cloned().collect::<Vec<_>>();
    Tensor::from_vec(data, shape, &candle_core::Device::Cpu).map_err(Into::into)
}

/// Copies a Candle tensor into an owned ndarray.
pub fn to_array<T>(tensor: &Tensor) -> TensorResult<ArrayD<T>>
where
    T: WithDType + Clone,
{
    let shape = tensor.dims().to_vec();
    let data = tensor.flatten_all()?.to_vec1::<T>()?;
    ArrayD::from_shape_vec(shape, data).map_err(Into::into)
}
