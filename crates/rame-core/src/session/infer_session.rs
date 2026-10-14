use crate::RameResult;
use crate::tensor::TensorMap;

pub trait InferSession {
    fn run(&mut self, inputs: TensorMap) -> RameResult<TensorMap>;
}
