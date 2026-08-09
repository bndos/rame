use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rame::session::ort::{
    OrtArenaExtendStrategy, OrtCudaExecutionProviderConfig, OrtSessionConfig,
};

use super::PyOrtCudaConfig;

impl TryFrom<&PyOrtCudaConfig> for OrtCudaExecutionProviderConfig {
    type Error = PyErr;

    fn try_from(config: &PyOrtCudaConfig) -> Result<Self, Self::Error> {
        let mut ep = OrtCudaExecutionProviderConfig::default().device_id(config.device_id);
        if let Some(limit) = config.memory_limit {
            ep = ep.memory_limit(limit);
        }
        if let Some(strategy) = config.arena_extend_strategy.as_deref() {
            ep = ep.arena_extend_strategy(parse_arena_extend_strategy(strategy)?);
        }
        if let Some(enable) = config.conv_max_workspace {
            ep = ep.conv_max_workspace(enable);
        }
        if let Some(enable) = config.tf32 {
            ep = ep.tf32(enable);
        }
        if let Some(enable) = config.prefer_nhwc {
            ep = ep.prefer_nhwc(enable);
        }
        Ok(ep)
    }
}

impl PyOrtCudaConfig {
    pub(super) fn add_to_session_config(
        &self,
        config: OrtSessionConfig,
    ) -> PyResult<OrtSessionConfig> {
        Ok(config.cuda_config(OrtCudaExecutionProviderConfig::try_from(self)?))
    }
}

fn parse_arena_extend_strategy(strategy: &str) -> PyResult<OrtArenaExtendStrategy> {
    match strategy {
        "next_power_of_two" => Ok(OrtArenaExtendStrategy::NextPowerOfTwo),
        "same_as_requested" => Ok(OrtArenaExtendStrategy::SameAsRequested),
        value => Err(PyValueError::new_err(format!(
            "unsupported ONNX Runtime arena extend strategy '{value}'"
        ))),
    }
}
