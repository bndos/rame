use crate::RameResult;
use crate::models::ModelError;
use crate::tensor::{DType, Device, Tensor, TensorError};

use super::{TdtGreedySearch, TdtHypothesis, TdtJointOutput};

/// Predicts a text-history representation from tensor-resident emitted tokens.
pub trait TensorPredictionNetwork {
    type State;
    type Output;

    fn initial_tensor_state(
        &mut self,
        batch_size: usize,
        device: &Device,
    ) -> RameResult<Self::State>;

    fn predict_tensor(
        &mut self,
        tokens: &Tensor,
        state: &Self::State,
    ) -> RameResult<(Self::Output, Self::State)>;

    fn replace_tensor_prediction(
        &mut self,
        output: &mut Self::Output,
        state: &mut Self::State,
        candidate_output: Self::Output,
        candidate_state: Self::State,
        replace_mask: &Tensor,
    ) -> RameResult<()>;
}

/// Combines tensor-resident acoustic and text-history representations.
pub trait TensorJointNetwork {
    type Encoded;
    type Predicted;
    type Output;

    fn joint_tensor(
        &mut self,
        encoded: &Self::Encoded,
        time_indices: &Tensor,
        predicted: &Self::Predicted,
    ) -> RameResult<Self::Output>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdtWorkspaceProfile {
    pub batch_size: usize,
    pub max_encoded_frames: usize,
    pub max_output_tokens: usize,
}

impl TdtWorkspaceProfile {
    pub fn new(batch_size: usize, max_encoded_frames: usize, max_output_tokens: usize) -> Self {
        Self {
            batch_size,
            max_encoded_frames,
            max_output_tokens,
        }
    }

    pub fn checked_output_slots(self) -> RameResult<usize> {
        self.batch_size.checked_mul(self.max_output_tokens).ok_or(
            ModelError::UnsupportedFeature {
                feature: "tensor TDT workspace",
                reason: "output token capacity overflows usize",
            }
            .into(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct TensorTdtWorkspace {
    profile: TdtWorkspaceProfile,
    device: Device,
    encoded_lengths: Tensor,
    time_indices: Tensor,
    safe_time_indices: Tensor,
    active_mask: Tensor,
    seeking_mask: Tensor,
    labels: Tensor,
    label_times: Tensor,
    label_durations: Tensor,
    found_label_mask: Tensor,
    symbols_at_time: Tensor,
    last_symbol_times: Tensor,
    token_decisions: Tensor,
    duration_decisions: Tensor,
    output_tokens: Tensor,
    output_start_frames: Tensor,
    output_durations: Tensor,
    output_counts: Tensor,
    completion_status: Tensor,
}

impl TensorTdtWorkspace {
    pub fn new(profile: TdtWorkspaceProfile, device: Device) -> RameResult<Self> {
        let batch = profile.batch_size;
        let output_slots = profile.checked_output_slots()?;

        Ok(Self {
            profile,
            encoded_lengths: zeros(batch, DType::I64, &device)?,
            time_indices: zeros(batch, DType::I64, &device)?,
            safe_time_indices: zeros(batch, DType::I64, &device)?,
            active_mask: zeros(batch, DType::U8, &device)?,
            seeking_mask: zeros(batch, DType::U8, &device)?,
            labels: zeros(batch, DType::I64, &device)?,
            label_times: zeros(batch, DType::I64, &device)?,
            label_durations: zeros(batch, DType::I64, &device)?,
            found_label_mask: zeros(batch, DType::U8, &device)?,
            symbols_at_time: zeros(batch, DType::I64, &device)?,
            last_symbol_times: zeros(batch, DType::I64, &device)?,
            token_decisions: zeros(batch, DType::I64, &device)?,
            duration_decisions: zeros(batch, DType::I64, &device)?,
            output_tokens: zeros(output_slots, DType::I64, &device)?,
            output_start_frames: zeros(output_slots, DType::I64, &device)?,
            output_durations: zeros(output_slots, DType::I64, &device)?,
            output_counts: zeros(batch, DType::I64, &device)?,
            completion_status: zeros(1, DType::U8, &device)?,
            device,
        })
    }

    pub fn profile(&self) -> TdtWorkspaceProfile {
        self.profile
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn encoded_lengths(&self) -> &Tensor {
        &self.encoded_lengths
    }

    pub fn time_indices(&self) -> &Tensor {
        &self.time_indices
    }

    pub fn safe_time_indices(&self) -> &Tensor {
        &self.safe_time_indices
    }

    pub fn active_mask(&self) -> &Tensor {
        &self.active_mask
    }

    pub fn seeking_mask(&self) -> &Tensor {
        &self.seeking_mask
    }

    pub fn labels(&self) -> &Tensor {
        &self.labels
    }

    pub fn label_times(&self) -> &Tensor {
        &self.label_times
    }

    pub fn label_durations(&self) -> &Tensor {
        &self.label_durations
    }

    pub fn found_label_mask(&self) -> &Tensor {
        &self.found_label_mask
    }

    pub fn symbols_at_time(&self) -> &Tensor {
        &self.symbols_at_time
    }

    pub fn last_symbol_times(&self) -> &Tensor {
        &self.last_symbol_times
    }

    pub fn token_decisions(&self) -> &Tensor {
        &self.token_decisions
    }

    pub fn duration_decisions(&self) -> &Tensor {
        &self.duration_decisions
    }

    pub fn output_tokens(&self) -> &Tensor {
        &self.output_tokens
    }

    pub fn output_start_frames(&self) -> &Tensor {
        &self.output_start_frames
    }

    pub fn output_durations(&self) -> &Tensor {
        &self.output_durations
    }

    pub fn output_counts(&self) -> &Tensor {
        &self.output_counts
    }

    pub fn completion_status(&self) -> &Tensor {
        &self.completion_status
    }
}

#[derive(Debug, Clone)]
pub struct TensorTdtGreedyExecutor {
    device: Device,
    workspace: Option<TensorTdtWorkspace>,
}

impl TensorTdtGreedyExecutor {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            workspace: None,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn workspace(&self) -> Option<&TensorTdtWorkspace> {
        self.workspace.as_ref()
    }

    pub fn ensure_workspace(
        &mut self,
        profile: TdtWorkspaceProfile,
    ) -> RameResult<&mut TensorTdtWorkspace> {
        let reuse = self
            .workspace
            .as_ref()
            .is_some_and(|workspace| workspace.profile() == profile);
        if !reuse {
            self.workspace = Some(TensorTdtWorkspace::new(profile, self.device.clone())?);
        }
        self.workspace.as_mut().ok_or(
            ModelError::UnsupportedFeature {
                feature: "tensor TDT workspace",
                reason: "workspace initialization failed",
            }
            .into(),
        )
    }
}

impl TdtGreedySearch<TensorTdtGreedyExecutor> {
    pub fn decode_many_tensor<P, J>(
        &mut self,
        _encoded: &J::Encoded,
        encoded_lengths: &Tensor,
        _predictor: &mut P,
        _joint: &mut J,
    ) -> RameResult<Vec<TdtHypothesis>>
    where
        P: TensorPredictionNetwork,
        J: TensorJointNetwork<Predicted = P::Output, Output = TdtJointOutput>,
    {
        let batch_size = encoded_lengths.dims().first().copied().ok_or_else(|| {
            ModelError::InvalidTensorShape {
                name: "TDT encoded lengths".into(),
                expected: "rank 1 tensor".into(),
                actual: encoded_lengths.dims().to_vec(),
            }
        })?;
        if encoded_lengths.dims() != [batch_size] {
            return Err(ModelError::InvalidTensorShape {
                name: "TDT encoded lengths".into(),
                expected: "rank 1 tensor".into(),
                actual: encoded_lengths.dims().to_vec(),
            }
            .into());
        }

        self.executor_mut()
            .ensure_workspace(TdtWorkspaceProfile::new(batch_size, 0, 0))?;

        Err(ModelError::UnsupportedFeature {
            feature: "tensor TDT greedy decoding",
            reason: "tensor-resident loop is scaffolded but not implemented yet",
        }
        .into())
    }
}

fn zeros(len: usize, dtype: DType, device: &Device) -> RameResult<Tensor> {
    Tensor::zeros(len, dtype, device)
        .map_err(TensorError::from)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use crate::RameError;
    use crate::models::ModelError;
    use crate::runtime::transducer::tdt::{
        TdtDecodingConfig, TdtGreedySearch, TdtJointOutput, TdtWorkspaceProfile,
        TensorJointNetwork, TensorPredictionNetwork, TensorTdtGreedyExecutor, TensorTdtWorkspace,
    };
    use crate::tensor::{Device, Tensor};
    use crate::{RameResult, tensor::DType};

    struct Predictor;

    impl TensorPredictionNetwork for Predictor {
        type State = ();
        type Output = ();

        fn initial_tensor_state(
            &mut self,
            _batch_size: usize,
            _device: &Device,
        ) -> RameResult<Self::State> {
            Ok(())
        }

        fn predict_tensor(
            &mut self,
            _tokens: &Tensor,
            _state: &Self::State,
        ) -> RameResult<(Self::Output, Self::State)> {
            Ok(((), ()))
        }

        fn replace_tensor_prediction(
            &mut self,
            _output: &mut Self::Output,
            _state: &mut Self::State,
            _candidate_output: Self::Output,
            _candidate_state: Self::State,
            _replace_mask: &Tensor,
        ) -> RameResult<()> {
            Ok(())
        }
    }

    struct Joint;

    impl TensorJointNetwork for Joint {
        type Encoded = ();
        type Predicted = ();
        type Output = TdtJointOutput;

        fn joint_tensor(
            &mut self,
            _encoded: &Self::Encoded,
            _time_indices: &Tensor,
            _predicted: &Self::Predicted,
        ) -> RameResult<Self::Output> {
            unreachable!("tensor greedy scaffold should not call joint yet")
        }
    }

    #[test]
    fn tensor_workspace_allocates_profile_buffers_on_the_requested_device() {
        let workspace =
            TensorTdtWorkspace::new(TdtWorkspaceProfile::new(3, 12, 5), Device::Cpu).unwrap();

        assert_eq!(workspace.profile().batch_size, 3);
        assert_eq!(workspace.profile().max_encoded_frames, 12);
        assert_eq!(workspace.profile().max_output_tokens, 5);
        assert_eq!(workspace.time_indices().dims(), &[3]);
        assert_eq!(workspace.output_tokens().dims(), &[15]);
        assert_eq!(workspace.completion_status().dims(), &[1]);
    }

    #[test]
    fn tensor_greedy_entry_point_fails_explicitly_until_the_loop_is_implemented() {
        let config = TdtDecodingConfig::new(8_193, 8_192, [0, 1, 2, 3, 4], 10);
        let mut search =
            TdtGreedySearch::with_executor(config, TensorTdtGreedyExecutor::new(Device::Cpu));
        let encoded_lengths = Tensor::zeros(2, DType::I64, &Device::Cpu).unwrap();

        let error = search
            .decode_many_tensor(&(), &encoded_lengths, &mut Predictor, &mut Joint)
            .unwrap_err();

        assert!(matches!(
            error,
            RameError::Model(ModelError::UnsupportedFeature {
                feature: "tensor TDT greedy decoding",
                ..
            })
        ));
        assert_eq!(
            search.executor().workspace().unwrap().profile().batch_size,
            2
        );
    }
}
