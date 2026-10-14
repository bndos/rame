use std::collections::HashMap;

use candle_core::Storage;
use mlx_rs::Array;
use mlx_rs::error::Exception;

use crate::RameResult;
use crate::models::ModelError;
use crate::models::parakeet::ParakeetTdtConfig;
use crate::runtime::transducer::tdt::{TdtJointLayout, TdtJointOutput};
use crate::runtime::{JointNetwork, MaskedPredictionNetwork, PredictionNetwork};
use crate::tensor::{Device, Tensor, TensorError};
use crate::tokenization::TokenId;

struct LstmLayer {
    input_weight: Vec<f32>,
    hidden_weight: Vec<f32>,
    bias: Vec<f32>,
}

pub struct Predictor {
    config: ParakeetTdtConfig,
    embedding: Vec<f32>,
    lstm: Vec<LstmLayer>,
    projection_weight: Vec<f32>,
    projection_bias: Vec<f32>,
}

pub struct Joint {
    config: ParakeetTdtConfig,
    output_weight: Vec<f32>,
    output_bias: Vec<f32>,
    scratch: Vec<f32>,
}

pub struct PredictionState {
    hidden: Vec<f32>,
    cell: Vec<f32>,
}

pub struct TensorPredictor(Predictor);

pub struct TensorJoint {
    inner: Joint,
    layout: TdtJointLayout,
}

impl TensorPredictor {
    pub(super) fn load(
        weights: &mut HashMap<String, Array>,
        config: ParakeetTdtConfig,
    ) -> Result<Self, Exception> {
        Predictor::load(weights, config).map(Self)
    }
}

impl TensorJoint {
    pub(super) fn load(
        weights: &mut HashMap<String, Array>,
        layout: TdtJointLayout,
        config: ParakeetTdtConfig,
    ) -> Result<Self, Exception> {
        Ok(Self {
            inner: Joint::load(weights, config)?,
            layout,
        })
    }
}

impl Predictor {
    pub(super) fn load(
        weights: &mut HashMap<String, Array>,
        config: ParakeetTdtConfig,
    ) -> Result<Self, Exception> {
        let hidden = config.network.prediction_hidden;
        let joint_hidden = config.network.joint_hidden;
        let vocabulary = config.num_token_classes;
        let predictor = Self {
            config,
            embedding: take_f32(weights, "decoder.prediction.embed.weight")?,
            lstm: (0..config.network.prediction_layers)
                .map(|index| LstmLayer::load(weights, index, hidden))
                .collect::<Result<Vec<_>, _>>()?,
            projection_weight: take_f32(weights, "joint.pred.weight")?,
            projection_bias: take_f32(weights, "joint.pred.bias")?,
        };
        for (name, actual, wanted) in [
            ("embedding", predictor.embedding.len(), vocabulary * hidden),
            (
                "projection weight",
                predictor.projection_weight.len(),
                joint_hidden * hidden,
            ),
            (
                "projection bias",
                predictor.projection_bias.len(),
                joint_hidden,
            ),
        ] {
            if actual != wanted {
                return Err(Exception::custom(format!(
                    "invalid Parakeet {name} length: expected {wanted}, got {actual}"
                )));
            }
        }
        Ok(predictor)
    }
}

impl PredictionNetwork for Predictor {
    type Tokens = Option<TokenId>;
    type Output = Vec<f32>;
    type State = PredictionState;

    fn initial_state(&mut self, batch_size: usize) -> RameResult<Self::State> {
        if batch_size != 1 {
            return Err(ModelError::UnsupportedFeature {
                feature: "MLX TDT prediction batch",
                reason: "the CPU prediction network processes one row at a time",
            }
            .into());
        }
        Ok(PredictionState {
            hidden: vec![0.0; self.lstm.len() * self.config.network.prediction_hidden],
            cell: vec![0.0; self.lstm.len() * self.config.network.prediction_hidden],
        })
    }

    fn predict(
        &mut self,
        token: &Self::Tokens,
        state: &Self::State,
    ) -> RameResult<(Self::Output, Self::State)> {
        let hidden = self.config.network.prediction_hidden;
        let gates_count = 4 * hidden;
        let vocabulary = self.config.num_token_classes;
        if token.is_some_and(|token| token as usize >= vocabulary) {
            return Err(ModelError::InvalidInput(
                "Parakeet prediction token is out of bounds".to_string(),
            )
            .into());
        }
        let mut value = token.map_or_else(
            || vec![0.0; hidden],
            |token| {
                let start = token as usize * hidden;
                self.embedding[start..start + hidden].to_vec()
            },
        );
        let mut next_hidden = vec![0.0; self.lstm.len() * hidden];
        let mut next_cell = vec![0.0; self.lstm.len() * hidden];
        let mut gates = vec![0.0; gates_count];
        for (index, layer) in self.lstm.iter().enumerate() {
            let offset = index * hidden;
            gemv(
                &layer.input_weight,
                gates_count,
                hidden,
                &value,
                &layer.bias,
                &mut gates,
            );
            gemv_accumulate(
                &layer.hidden_weight,
                gates_count,
                hidden,
                &state.hidden[offset..offset + hidden],
                &mut gates,
            );
            for unit in 0..hidden {
                let input_gate = sigmoid(gates[unit]);
                let forget_gate = sigmoid(gates[hidden + unit]);
                let candidate = gates[2 * hidden + unit].tanh();
                let output_gate = sigmoid(gates[3 * hidden + unit]);
                let next = forget_gate * state.cell[offset + unit] + input_gate * candidate;
                next_cell[offset + unit] = next;
                next_hidden[offset + unit] = output_gate * next.tanh();
            }
            value.copy_from_slice(&next_hidden[offset..offset + hidden]);
        }
        let joint_hidden = self.config.network.joint_hidden;
        let mut projection = vec![0.0; joint_hidden];
        gemv(
            &self.projection_weight,
            joint_hidden,
            hidden,
            &value,
            &self.projection_bias,
            &mut projection,
        );
        Ok((
            projection,
            PredictionState {
                hidden: next_hidden,
                cell: next_cell,
            },
        ))
    }
}

impl PredictionNetwork for TensorPredictor {
    type Tokens = Tensor;
    type Output = Tensor;
    type State = Vec<PredictionState>;

    fn initial_state(&mut self, batch_size: usize) -> RameResult<Self::State> {
        (0..batch_size).map(|_| self.0.initial_state(1)).collect()
    }

    fn predict(
        &mut self,
        tokens: &Tensor,
        state: &Self::State,
    ) -> RameResult<(Tensor, Self::State)> {
        let labels = tokens.to_vec1::<i64>().map_err(TensorError::from)?;
        if labels.len() != state.len() {
            return Err(crate::RameError::InvalidBatchLength {
                stage: "Parakeet prediction state",
                expected: labels.len(),
                actual: state.len(),
            });
        }
        let hidden = self.0.config.network.joint_hidden;
        let mut outputs = Vec::with_capacity(labels.len() * hidden);
        let mut next_states = Vec::with_capacity(labels.len());
        for (label, state) in labels.into_iter().zip(state) {
            let token = if label == self.0.config.blank_token_id as i64 {
                None
            } else {
                Some(u32::try_from(label).map_err(|_| {
                    ModelError::InvalidInput("Parakeet prediction token is out of bounds".into())
                })?)
            };
            let (output, next_state) = self.0.predict(&token, state)?;
            outputs.extend(output);
            next_states.push(next_state);
        }
        Ok((
            Tensor::from_vec(outputs, (state.len(), hidden), &Device::Cpu)
                .map_err(TensorError::from)?,
            next_states,
        ))
    }
}

impl MaskedPredictionNetwork for TensorPredictor {
    type Mask = Tensor;

    fn replace_prediction(
        &mut self,
        output: &mut Tensor,
        state: &mut Self::State,
        candidate_output: Tensor,
        candidate_state: Self::State,
        replace_mask: &Tensor,
    ) -> RameResult<()> {
        let mask = replace_mask.to_vec1::<u8>().map_err(TensorError::from)?;
        if mask.len() != state.len() || candidate_state.len() != state.len() {
            return Err(
                ModelError::InvalidInput("Parakeet prediction mask batch mismatch".into()).into(),
            );
        }
        *output = replace_mask
            .unsqueeze(1)
            .and_then(|mask| mask.broadcast_as(output.shape()))
            .and_then(|mask| mask.where_cond(&candidate_output, output))
            .map_err(TensorError::from)?;
        for ((current, candidate), replace) in state.iter_mut().zip(candidate_state).zip(mask) {
            if replace != 0 {
                *current = candidate;
            }
        }
        Ok(())
    }
}

impl Joint {
    pub(super) fn load(
        weights: &mut HashMap<String, Array>,
        config: ParakeetTdtConfig,
    ) -> Result<Self, Exception> {
        let hidden = config.network.joint_hidden;
        let outputs = config.joint_logit_count();
        let output_weight = take_f32(weights, "joint.joint_net.2.weight")?;
        let output_bias = take_f32(weights, "joint.joint_net.2.bias")?;
        if output_weight.len() != outputs * hidden || output_bias.len() != outputs {
            return Err(Exception::custom("invalid Parakeet joint weight shape"));
        }
        Ok(Self {
            config,
            output_weight,
            output_bias,
            scratch: vec![0.0; hidden],
        })
    }
}

impl JointNetwork for Joint {
    type Encoded = Vec<f32>;
    type Indices = usize;
    type Predicted = Vec<f32>;
    type Output = Vec<f32>;

    fn joint(
        &mut self,
        encoded: &Self::Encoded,
        time_index: &Self::Indices,
        predicted: &Self::Predicted,
    ) -> RameResult<Self::Output> {
        self.joint_slices(encoded, *time_index, predicted)
    }
}

impl Joint {
    fn joint_slices(
        &mut self,
        encoded: &[f32],
        time_index: usize,
        predicted: &[f32],
    ) -> RameResult<Vec<f32>> {
        let hidden = self.config.network.joint_hidden;
        let outputs = self.config.joint_logit_count();
        if predicted.len() != hidden {
            return Err(ModelError::InvalidTensorShape {
                name: "Parakeet prediction".to_string(),
                expected: format!("{hidden} values"),
                actual: vec![predicted.len()],
            }
            .into());
        }
        let start = time_index.checked_mul(hidden).ok_or_else(|| {
            ModelError::InvalidInput("Parakeet encoded frame index overflow".to_string())
        })?;
        let end = start.checked_add(hidden).ok_or_else(|| {
            ModelError::InvalidInput("Parakeet encoded frame index overflow".to_string())
        })?;
        let frame = encoded.get(start..end).ok_or_else(|| {
            ModelError::InvalidInput("Parakeet encoded frame is out of bounds".to_string())
        })?;
        for ((output, encoded), predicted) in self.scratch.iter_mut().zip(frame).zip(predicted) {
            *output = (*encoded + *predicted).max(0.0);
        }
        let mut logits = vec![0.0; outputs];
        gemv(
            &self.output_weight,
            outputs,
            hidden,
            &self.scratch,
            &self.output_bias,
            &mut logits,
        );
        Ok(logits)
    }
}

impl JointNetwork for TensorJoint {
    type Encoded = Tensor;
    type Indices = Tensor;
    type Predicted = Tensor;
    type Output = TdtJointOutput;

    fn joint(
        &mut self,
        encoded: &Tensor,
        time_indices: &Tensor,
        predicted: &Tensor,
    ) -> RameResult<Self::Output> {
        let hidden = self.inner.config.network.joint_hidden;
        let [batch, frames, width] = encoded.dims() else {
            return Err(ModelError::InvalidTensorShape {
                name: "Parakeet encoded frames".into(),
                expected: "[batch, frames, width]".into(),
                actual: encoded.dims().to_vec(),
            }
            .into());
        };
        if *width != hidden
            || time_indices.dims() != [*batch]
            || predicted.dims() != [*batch, hidden]
        {
            return Err(ModelError::InvalidInput(
                "Parakeet joint tensor dimensions are invalid".into(),
            )
            .into());
        }
        let (encoded_storage, encoded_layout) = encoded.storage_and_layout();
        let (predicted_storage, predicted_layout) = predicted.storage_and_layout();
        if !encoded_layout.is_contiguous() || !predicted_layout.is_contiguous() {
            return Err(ModelError::InvalidInput(
                "Parakeet joint requires contiguous CPU tensors".into(),
            )
            .into());
        }
        let Storage::Cpu(encoded_storage) = &*encoded_storage else {
            return Err(ModelError::InvalidInput(
                "Parakeet joint requires CPU encoded frames".into(),
            )
            .into());
        };
        let Storage::Cpu(predicted_storage) = &*predicted_storage else {
            return Err(
                ModelError::InvalidInput("Parakeet joint requires CPU predictions".into()).into(),
            );
        };
        let encoded_values = encoded_storage
            .as_slice::<f32>()
            .map_err(TensorError::from)?;
        let predicted_values = predicted_storage
            .as_slice::<f32>()
            .map_err(TensorError::from)?;
        let encoded_values = &encoded_values[encoded_layout.start_offset()..];
        let predicted_values = &predicted_values[predicted_layout.start_offset()..];
        let indices = time_indices.to_vec1::<i64>().map_err(TensorError::from)?;
        let mut logits = Vec::with_capacity(batch * self.layout.joint_logit_count());
        for (row, index) in indices.into_iter().enumerate() {
            let index = usize::try_from(index)
                .map_err(|_| ModelError::InvalidInput("Parakeet frame index is negative".into()))?;
            if index >= *frames {
                return Err(ModelError::InvalidInput(
                    "Parakeet frame index is out of bounds".into(),
                )
                .into());
            }
            let row_start = row * frames * hidden;
            let predicted_start = row * hidden;
            logits.extend(self.inner.joint_slices(
                &encoded_values[row_start..row_start + frames * hidden],
                index,
                &predicted_values[predicted_start..predicted_start + hidden],
            )?);
        }
        let logits = Tensor::from_vec(
            logits,
            (*batch, self.layout.joint_logit_count()),
            &Device::Cpu,
        )
        .map_err(TensorError::from)?;
        TdtJointOutput::from_combined(logits, self.layout)
    }
}

impl LstmLayer {
    fn load(
        weights: &mut HashMap<String, Array>,
        index: usize,
        hidden: usize,
    ) -> Result<Self, Exception> {
        let gates = 4 * hidden;
        let prefix = format!("decoder.prediction.dec_rnn.lstm.{index}");
        let layer = Self {
            input_weight: take_f32(weights, &format!("{prefix}.Wx"))?,
            hidden_weight: take_f32(weights, &format!("{prefix}.Wh"))?,
            bias: take_f32(weights, &format!("{prefix}.bias"))?,
        };
        for (name, actual, wanted) in [
            ("input weight", layer.input_weight.len(), gates * hidden),
            ("hidden weight", layer.hidden_weight.len(), gates * hidden),
            ("bias", layer.bias.len(), gates),
        ] {
            if actual != wanted {
                return Err(Exception::custom(format!(
                    "invalid Parakeet LSTM {index} {name} length: expected {wanted}, got {actual}"
                )));
            }
        }
        Ok(layer)
    }
}

fn take_f32(weights: &mut HashMap<String, Array>, name: &str) -> Result<Vec<f32>, Exception> {
    weights
        .remove(name)
        .ok_or_else(|| Exception::custom(format!("missing Parakeet MLX weight `{name}`")))?
        .to_vec_cast::<f32>()
        .map_err(|error| Exception::custom(error.to_string()))
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn gemv(
    matrix: &[f32],
    rows: usize,
    columns: usize,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    output.copy_from_slice(bias);
    gemv_accumulate(matrix, rows, columns, input, output);
}

fn gemv_accumulate(matrix: &[f32], rows: usize, columns: usize, input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(matrix.len(), rows * columns);
    debug_assert_eq!(input.len(), columns);
    debug_assert_eq!(output.len(), rows);
    // The validated slices remain alive for this synchronous BLAS call.
    unsafe {
        cblas_sgemv(
            101,
            111,
            rows as i32,
            columns as i32,
            1.0,
            matrix.as_ptr(),
            columns as i32,
            input.as_ptr(),
            1,
            1.0,
            output.as_mut_ptr(),
            1,
        );
    }
}

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemv(
        order: i32,
        transpose: i32,
        rows: i32,
        columns: i32,
        alpha: f32,
        matrix: *const f32,
        leading_dimension: i32,
        input: *const f32,
        input_stride: i32,
        beta: f32,
        output: *mut f32,
        output_stride: i32,
    );
}
