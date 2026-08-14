use std::path::Path;

use ort::session::{OutputSelector, RunOptions, Session};

use crate::RameResult;
use crate::session::ort::{OrtError, OrtSessionConfig};
use crate::session::{InferSession, SessionBackend};
use crate::tensor::TensorMap;

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
        let input_buffers = inputs.into_iter().collect::<Vec<_>>();
        let input_views = input_buffers
            .iter()
            .map(|(name, tensor)| tensor.ort_input(name))
            .collect::<Result<Vec<_>, _>>()
            .map_err(OrtError::from)?;
        let inputs = input_views
            .iter()
            .map(|input| input.as_session_input())
            .collect::<Result<Vec<_>, _>>()
            .map_err(OrtError::from)?;

        let tensors = if self.output_names.is_empty() {
            let outputs = self.session.run(inputs).map_err(OrtError::from)?;
            outputs.try_into()?
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
            outputs.try_into()?
        };

        Ok(tensors)
    }
}
