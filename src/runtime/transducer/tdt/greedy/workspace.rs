use crate::models::ModelError;
use crate::runtime::transducer::tdt::{TdtDecodingConfig, TdtHypothesis, TdtTokenAlignment};
use crate::tensor::{DType, Device, Tensor, TensorError, Var};
use crate::{RameResult, tokenization::TokenId};

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkspaceProfile {
    batch_size: usize,
    max_encoded_frames: usize,
    max_output_tokens: usize,
}

impl WorkspaceProfile {
    fn new(
        batch_size: usize,
        max_encoded_frames: usize,
        config: &TdtDecodingConfig,
    ) -> RameResult<Self> {
        let max_output_tokens = output_capacity(max_encoded_frames, config.max_symbols_per_step())?;
        Ok(Self {
            batch_size,
            max_encoded_frames,
            max_output_tokens,
        })
    }
}

#[derive(Debug)]
pub(super) struct Workspace {
    profile: WorkspaceProfile,
    durations: Tensor,
    zeros: Tensor,
    ones: Tensor,
    blank_labels: Tensor,
    initial_symbol_times: Tensor,
    pub(super) encoded_lengths: Tensor,
    pub(super) time_indices: Tensor,
    pub(super) labels: Tensor,
    pub(super) label_times: Tensor,
    pub(super) label_durations: Tensor,
    pub(super) symbols_at_time: Tensor,
    pub(super) last_symbol_times: Tensor,
    output_tokens: Var,
    output_start_frames: Var,
    output_durations: Var,
    output_counts: Tensor,
}

impl Workspace {
    pub(super) fn new(
        device: &Device,
        encoded_lengths: &Tensor,
        max_encoded_frames: usize,
        config: &TdtDecodingConfig,
    ) -> RameResult<Self> {
        let batch_size = encoded_lengths.elem_count();
        let profile = WorkspaceProfile::new(batch_size, max_encoded_frames, config)?;
        let encoded_lengths = encoded_lengths
            .to_dtype(DType::I64)
            .map_err(TensorError::from)?;
        let zeros = Tensor::zeros(batch_size, DType::I64, device).map_err(TensorError::from)?;
        let ones = Tensor::ones(batch_size, DType::I64, device).map_err(TensorError::from)?;
        let blank_labels = Tensor::full(i64::from(config.blank_token_id()), batch_size, device)
            .map_err(TensorError::from)?;
        let initial_symbol_times = (&zeros - &ones).map_err(TensorError::from)?;
        let durations = Tensor::from_vec(
            config
                .durations()
                .iter()
                .map(|&duration| duration as i64)
                .collect::<Vec<_>>(),
            config.durations().len(),
            device,
        )
        .map_err(TensorError::from)?;
        let output_shape = (batch_size, profile.max_output_tokens);

        Ok(Self {
            profile,
            durations,
            encoded_lengths,
            time_indices: zeros.clone(),
            labels: blank_labels.clone(),
            label_times: zeros.clone(),
            label_durations: zeros.clone(),
            symbols_at_time: zeros.clone(),
            last_symbol_times: initial_symbol_times.clone(),
            output_tokens: Var::zeros(output_shape, DType::I64, device)
                .map_err(TensorError::from)?,
            output_start_frames: Var::zeros(output_shape, DType::I64, device)
                .map_err(TensorError::from)?,
            output_durations: Var::zeros(output_shape, DType::I64, device)
                .map_err(TensorError::from)?,
            output_counts: zeros.clone(),
            zeros,
            ones,
            blank_labels,
            initial_symbol_times,
        })
    }

    pub(super) fn is_compatible(
        &self,
        device: &Device,
        encoded_lengths: &Tensor,
        max_encoded_frames: usize,
        max_symbols_per_step: usize,
    ) -> RameResult<bool> {
        let requested_output_tokens = output_capacity(max_encoded_frames, max_symbols_per_step)?;
        Ok(self.profile.batch_size == encoded_lengths.elem_count()
            && self.profile.max_encoded_frames >= max_encoded_frames
            && self.profile.max_output_tokens >= requested_output_tokens
            && self.encoded_lengths.device().same_device(device))
    }

    pub(super) fn reset(&mut self, encoded_lengths: &Tensor) -> RameResult<()> {
        self.encoded_lengths = encoded_lengths
            .to_dtype(DType::I64)
            .map_err(TensorError::from)?;
        self.time_indices = self.zeros.clone();
        self.labels = self.blank_labels.clone();
        self.label_times = self.zeros.clone();
        self.label_durations = self.zeros.clone();
        self.symbols_at_time = self.zeros.clone();
        self.last_symbol_times = self.initial_symbol_times.clone();
        self.output_tokens.zero_set().map_err(TensorError::from)?;
        self.output_start_frames
            .zero_set()
            .map_err(TensorError::from)?;
        self.output_durations
            .zero_set()
            .map_err(TensorError::from)?;
        self.output_counts = self.zeros.clone();
        Ok(())
    }

    pub(super) fn zeros(&self) -> &Tensor {
        &self.zeros
    }

    pub(super) fn ones(&self) -> &Tensor {
        &self.ones
    }

    pub(super) fn blank_labels(&self) -> &Tensor {
        &self.blank_labels
    }

    pub(super) fn durations(&self) -> &Tensor {
        &self.durations
    }

    pub(super) fn record(&mut self, found: &Tensor) -> RameResult<()> {
        let last_slot = (self.profile.max_output_tokens - 1) as i64;
        let safe_counts = self
            .output_counts
            .clamp(0i64, last_slot)
            .map_err(TensorError::from)?;
        let indices = safe_counts
            .unsqueeze(1)
            .and_then(|indices| indices.to_dtype(DType::U32))
            .map_err(TensorError::from)?;
        let mask = found.unsqueeze(1).map_err(TensorError::from)?;

        masked_scatter(&self.output_tokens, &indices, &self.labels, &mask)?;
        masked_scatter(
            &self.output_start_frames,
            &indices,
            &self.label_times,
            &mask,
        )?;
        masked_scatter(
            &self.output_durations,
            &indices,
            &self.label_durations,
            &mask,
        )?;
        self.output_counts = (&self.output_counts
            + &found.to_dtype(DType::I64).map_err(TensorError::from)?)
            .map_err(TensorError::from)?;
        Ok(())
    }

    pub(super) fn hypotheses(&self) -> RameResult<Vec<TdtHypothesis>> {
        let counts = self
            .output_counts
            .to_device(&Device::Cpu)
            .and_then(|counts| counts.to_vec1::<i64>())
            .map_err(TensorError::from)?;
        let max_count = counts.iter().copied().max().unwrap_or(0) as usize;
        let mut hypotheses = counts
            .iter()
            .map(|_| TdtHypothesis::new(Vec::new()))
            .collect::<Vec<_>>();
        if max_count == 0 {
            return Ok(hypotheses);
        }

        let tokens = copy_output(&self.output_tokens, max_count)?;
        let start_frames = copy_output(&self.output_start_frames, max_count)?;
        let durations = copy_output(&self.output_durations, max_count)?;
        for batch_index in 0..self.profile.batch_size {
            for output_index in 0..counts[batch_index] as usize {
                hypotheses[batch_index].push(TdtTokenAlignment::new(
                    tokens[batch_index][output_index] as TokenId,
                    start_frames[batch_index][output_index] as usize,
                    durations[batch_index][output_index] as usize,
                ));
            }
        }
        Ok(hypotheses)
    }
}

fn output_capacity(max_encoded_frames: usize, max_symbols_per_step: usize) -> RameResult<usize> {
    max_encoded_frames
        .checked_mul(max_symbols_per_step.max(1))
        .ok_or_else(|| {
            ModelError::UnsupportedFeature {
                feature: "standard TDT greedy workspace",
                reason: "output token capacity overflows usize",
            }
            .into()
        })
}

fn masked_scatter(
    destination: &Tensor,
    indices: &Tensor,
    values: &Tensor,
    mask: &Tensor,
) -> RameResult<()> {
    let current = destination.gather(indices, 1).map_err(TensorError::from)?;
    let values = values.unsqueeze(1).map_err(TensorError::from)?;
    let source = mask
        .where_cond(&values, &current)
        .map_err(TensorError::from)?;
    destination
        .scatter_set(indices, &source, 1)
        .map_err(TensorError::from)
        .map_err(Into::into)
}

fn copy_output(output: &Tensor, length: usize) -> RameResult<Vec<Vec<i64>>> {
    output
        .narrow(1, 0, length)
        .and_then(|output| output.to_device(&Device::Cpu))
        .and_then(|output| output.to_vec2::<i64>())
        .map_err(TensorError::from)
        .map_err(Into::into)
}
