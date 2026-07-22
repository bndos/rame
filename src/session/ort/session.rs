use std::path::Path;

use ort::session::Session;
use ort::session::SessionInputValue;
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
        let builder = Session::builder().map_err(OrtError::from)?;
        let mut builder = config.apply(builder)?;

        let session = builder.commit_from_file(path).map_err(OrtError::from)?;

        Ok(OrtSession { session })
    }
}

#[derive(Debug)]
pub struct OrtSession {
    session: Session,
}

impl InferSession for OrtSession {
    fn run(&mut self, inputs: TensorMap) -> RameResult<TensorMap> {
        let inputs: Vec<(String, SessionInputValue<'_>)> = inputs
            .into_iter()
            .map(|(name, value)| match value {
                TensorValue::F32(array) => {
                    Tensor::from_array(array).map(|tensor| (name, tensor.into()))
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(OrtError::from)?;

        let outputs = self.session.run(inputs).map_err(OrtError::from)?;
        let mut tensors = TensorMap::new();
        for (name, value) in outputs {
            let value = match value.dtype().tensor_type() {
                Some(TensorElementType::Float32) => TensorValue::F32(
                    value
                        .try_extract_array::<f32>()
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
}
