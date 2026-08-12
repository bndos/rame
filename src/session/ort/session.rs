use std::path::Path;

use ort::session::SessionInputValue;
use ort::session::{OutputSelector, RunOptions, Session, SessionOutputs};
use ort::value::{Tensor, TensorElementType};

use crate::RameResult;
use crate::session::ort::{OrtError, OrtSessionConfig};
use crate::session::{InferSession, SessionBackend};
use crate::tensor::{TensorMap, TensorValue};

#[derive(Debug, Clone, Copy)]
pub struct OrtBackend;

impl SessionBackend for OrtBackend {
    type Config = OrtSessionConfig;
    type Session = OrtSession;

    fn load(path: &Path, config: Self::Config) -> RameResult<Self::Session> {
        let output_names = config.output_names.clone();
        let builder = Session::builder().map_err(OrtError::from)?;
        let mut builder = config.apply(builder)?;

        let session = builder.commit_from_file(path).map_err(OrtError::from)?;

        Ok(OrtSession {
            session,
            output_names,
        })
    }
}

#[derive(Debug)]
pub struct OrtSession {
    session: Session,
    output_names: Vec<String>,
}

impl InferSession for OrtSession {
    fn run(&mut self, inputs: TensorMap) -> RameResult<TensorMap> {
        let inputs: Vec<(String, SessionInputValue<'_>)> = inputs
            .into_iter()
            .map(|(name, value)| match value {
                TensorValue::F32(array) => {
                    Tensor::from_array(array).map(|tensor| (name, tensor.into()))
                }
                TensorValue::I32(array) => {
                    Tensor::from_array(array).map(|tensor| (name, tensor.into()))
                }
                TensorValue::I64(array) => {
                    Tensor::from_array(array).map(|tensor| (name, tensor.into()))
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(OrtError::from)?;

        let tensors = if self.output_names.is_empty() {
            let outputs = self.session.run(inputs).map_err(OrtError::from)?;
            outputs_to_tensor_map(outputs)?
        } else {
            let selector = self
                .output_names
                .iter()
                .fold(OutputSelector::no_default(), |selector, name| {
                    selector.with(name)
                });
            let options = RunOptions::new()
                .map_err(OrtError::from)?
                .with_outputs(selector);
            let outputs = self
                .session
                .run_with_options(inputs, &options)
                .map_err(OrtError::from)?;
            outputs_to_tensor_map(outputs)?
        };

        Ok(tensors)
    }
}

fn outputs_to_tensor_map(outputs: SessionOutputs<'_>) -> RameResult<TensorMap> {
    let mut tensors = TensorMap::new();
    for (name, value) in outputs {
        let value = match value.dtype().tensor_type() {
            Some(TensorElementType::Float32) => TensorValue::F32(
                value
                    .try_extract_array::<f32>()
                    .map_err(OrtError::from)?
                    .to_owned(),
            ),
            Some(TensorElementType::Int32) => TensorValue::I32(
                value
                    .try_extract_array::<i32>()
                    .map_err(OrtError::from)?
                    .to_owned(),
            ),
            Some(TensorElementType::Int64) => TensorValue::I64(
                value
                    .try_extract_array::<i64>()
                    .map_err(OrtError::from)?
                    .to_owned(),
            ),
            Some(tensor_type) => {
                return Err(OrtError::UnsupportedTensorType {
                    name: name.to_string(),
                    tensor_type: tensor_type.to_string(),
                }
                .into());
            }
            None => {
                return Err(OrtError::UnsupportedTensorType {
                    name: name.to_string(),
                    tensor_type: value.dtype().to_string(),
                }
                .into());
            }
        };

        tensors.insert(name.to_string(), value);
    }
    Ok(tensors)
}
