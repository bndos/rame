use std::collections::HashMap;

use mlx_rs::error::Exception;
use mlx_rs::ops::{broadcast_to, select, sigmoid, tanh};
use mlx_rs::{Array, Dtype};

use crate::RameResult;
use crate::models::ModelError;
use crate::models::parakeet::ParakeetTdtConfig;
use crate::runtime::{JointNetwork, MaskedPredictionNetwork, PredictionNetwork};

struct LstmLayer {
    input_weight: Array,
    hidden_weight: Array,
    bias: Array,
}

pub struct Predictor {
    config: ParakeetTdtConfig,
    embedding: Array,
    lstm: Vec<LstmLayer>,
    projection_weight: Array,
    projection_bias: Array,
}

pub struct PredictionState {
    hidden: Vec<Array>,
    cell: Vec<Array>,
}

pub struct Joint {
    config: ParakeetTdtConfig,
    output_weight: Array,
    output_bias: Array,
}

impl Predictor {
    pub(super) fn load(
        weights: &mut HashMap<String, Array>,
        config: ParakeetTdtConfig,
    ) -> Result<Self, Exception> {
        let hidden = dimension(config.network.prediction_hidden)?;
        let joint_hidden = dimension(config.network.joint_hidden)?;
        let vocabulary = dimension(config.num_token_classes)?;
        let embedding =
            take(weights, "decoder.prediction.embed.weight")?.reshape(&[vocabulary, hidden])?;
        let lstm = (0..config.network.prediction_layers)
            .map(|index| LstmLayer::load(weights, index, hidden))
            .collect::<Result<Vec<_>, _>>()?;
        let projection_weight = take(weights, "joint.pred.weight")?
            .reshape(&[joint_hidden, hidden])?
            .transpose()?;
        let projection_bias = take(weights, "joint.pred.bias")?.reshape(&[joint_hidden])?;
        Ok(Self {
            config,
            embedding,
            lstm,
            projection_weight,
            projection_bias,
        })
    }
}

impl LstmLayer {
    fn load(
        weights: &mut HashMap<String, Array>,
        index: usize,
        hidden: i32,
    ) -> Result<Self, Exception> {
        let gates = hidden
            .checked_mul(4)
            .ok_or_else(|| Exception::custom("Parakeet LSTM width exceeds i32"))?;
        let prefix = format!("decoder.prediction.dec_rnn.lstm.{index}");
        Ok(Self {
            input_weight: take(weights, &format!("{prefix}.Wx"))?
                .reshape(&[gates, hidden])?
                .transpose()?,
            hidden_weight: take(weights, &format!("{prefix}.Wh"))?
                .reshape(&[gates, hidden])?
                .transpose()?,
            bias: take(weights, &format!("{prefix}.bias"))?.reshape(&[gates])?,
        })
    }
}

impl PredictionNetwork for Predictor {
    type Tokens = Array;
    type Output = Array;
    type State = PredictionState;

    fn initial_state(&mut self, batch_size: usize) -> RameResult<Self::State> {
        let batch = i32::try_from(batch_size)
            .map_err(|_| ModelError::InvalidInput("MLX prediction batch exceeds i32".into()))?;
        let hidden = dimension(self.config.network.prediction_hidden).map_err(mlx_error)?;
        let zero = Array::zeros::<f32>(&[batch, hidden]).map_err(mlx_error)?;
        Ok(PredictionState {
            hidden: vec![zero.clone(); self.lstm.len()],
            cell: vec![zero; self.lstm.len()],
        })
    }

    fn predict(&mut self, tokens: &Array, state: &Self::State) -> RameResult<(Array, Self::State)> {
        if tokens.ndim() != 1 {
            return Err(ModelError::InvalidTensorShape {
                name: "MLX prediction tokens".into(),
                expected: "[batch]".into(),
                actual: tokens.shape().iter().map(|&n| n as usize).collect(),
            }
            .into());
        }
        let tokens = tokens.as_dtype(Dtype::Int32).map_err(mlx_error)?;
        let embedded = self.embedding.take_axis(&tokens, 0).map_err(mlx_error)?;
        let blank = tokens
            .eq(Array::from_int(self.config.blank_token_id as i32))
            .map_err(mlx_error)?;
        let zeros = Array::zeros::<f32>(embedded.shape()).map_err(mlx_error)?;
        let mut value = select(
            &blank.expand_dims(-1).map_err(mlx_error)?,
            &zeros,
            &embedded,
        )
        .map_err(mlx_error)?;
        let mut next_hidden = Vec::with_capacity(self.lstm.len());
        let mut next_cell = Vec::with_capacity(self.lstm.len());
        for (index, layer) in self.lstm.iter().enumerate() {
            let gates = value.matmul(&layer.input_weight).map_err(mlx_error)?
                + state.hidden[index]
                    .matmul(&layer.hidden_weight)
                    .map_err(mlx_error)?
                + &layer.bias;
            let gates = gates.split_equal(4, -1).map_err(mlx_error)?;
            let input_gate = sigmoid(&gates[0]).map_err(mlx_error)?;
            let forget_gate = sigmoid(&gates[1]).map_err(mlx_error)?;
            let candidate = tanh(&gates[2]).map_err(mlx_error)?;
            let output_gate = sigmoid(&gates[3]).map_err(mlx_error)?;
            let cell = &forget_gate * &state.cell[index] + &input_gate * &candidate;
            let hidden = output_gate * tanh(&cell).map_err(mlx_error)?;
            value = hidden.clone();
            next_hidden.push(hidden);
            next_cell.push(cell);
        }
        let output =
            value.matmul(&self.projection_weight).map_err(mlx_error)? + &self.projection_bias;
        Ok((
            output,
            PredictionState {
                hidden: next_hidden,
                cell: next_cell,
            },
        ))
    }
}

impl MaskedPredictionNetwork for Predictor {
    type Mask = Array;

    fn replace_prediction(
        &mut self,
        output: &mut Array,
        state: &mut Self::State,
        candidate_output: Array,
        candidate_state: Self::State,
        replace_mask: &Array,
    ) -> RameResult<()> {
        let mask = replace_mask.expand_dims(-1).map_err(mlx_error)?;
        *output = select(&mask, &candidate_output, &*output).map_err(mlx_error)?;
        for index in 0..state.hidden.len() {
            state.hidden[index] =
                select(&mask, &candidate_state.hidden[index], &state.hidden[index])
                    .map_err(mlx_error)?;
            state.cell[index] = select(&mask, &candidate_state.cell[index], &state.cell[index])
                .map_err(mlx_error)?;
        }
        Ok(())
    }
}

impl Joint {
    pub(super) fn load(
        weights: &mut HashMap<String, Array>,
        config: ParakeetTdtConfig,
    ) -> Result<Self, Exception> {
        let hidden = dimension(config.network.joint_hidden)?;
        let outputs = dimension(config.joint_logit_count())?;
        Ok(Self {
            config,
            output_weight: take(weights, "joint.joint_net.2.weight")?
                .reshape(&[outputs, hidden])?
                .transpose()?,
            output_bias: take(weights, "joint.joint_net.2.bias")?.reshape(&[outputs])?,
        })
    }
}

impl JointNetwork for Joint {
    type Encoded = Array;
    type Indices = Array;
    type Predicted = Array;
    type Output = Array;

    fn joint(
        &mut self,
        encoded: &Array,
        time_indices: &Array,
        predicted: &Array,
    ) -> RameResult<Array> {
        let [batch, _, width] = encoded.shape() else {
            return Err(
                ModelError::InvalidInput("MLX joint expects [batch, time, width]".into()).into(),
            );
        };
        let hidden = dimension(self.config.network.joint_hidden).map_err(mlx_error)?;
        if *width != hidden || predicted.shape() != [*batch, hidden] {
            return Err(ModelError::InvalidInput(
                "MLX joint encoded or predicted width is invalid".into(),
            )
            .into());
        }
        let indices = broadcast_to(
            &time_indices.reshape(&[*batch, 1, 1]).map_err(mlx_error)?,
            &[*batch, 1, hidden],
        )
        .map_err(mlx_error)?;
        let frame = encoded
            .take_along_axis(&indices, 1)
            .map_err(mlx_error)?
            .squeeze_axes(&[1])
            .map_err(mlx_error)?;
        let sum = &frame + predicted;
        let relu = mlx_rs::ops::maximum(&sum, Array::from_f32(0.0)).map_err(mlx_error)?;
        Ok(relu.matmul(&self.output_weight).map_err(mlx_error)? + &self.output_bias)
    }
}

fn take(weights: &mut HashMap<String, Array>, name: &str) -> Result<Array, Exception> {
    weights
        .remove(name)
        .ok_or_else(|| Exception::custom(format!("missing Parakeet MLX weight `{name}`")))
}

fn dimension(value: usize) -> Result<i32, Exception> {
    i32::try_from(value).map_err(|_| Exception::custom("Parakeet dimension exceeds i32"))
}

fn mlx_error(error: Exception) -> crate::RameError {
    ModelError::InvalidInput(format!("Parakeet MLX decoder: {error}")).into()
}
