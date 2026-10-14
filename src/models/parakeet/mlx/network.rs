use std::collections::HashMap;

use half::bf16;
use mlx_rs::error::Exception;
use mlx_rs::fast::{ScaledDotProductAttentionMask, layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
use mlx_rs::ops::{arange, conv1d, conv2d, maximum, pad, quantized_matmul, sigmoid, stack};
use mlx_rs::{Array, Dtype};

use super::artifact::WeightFormat;

const WIDTH: i32 = 1024;
const HEADS: i32 = 8;
const HEAD_DIM: i32 = 128;

struct Linear {
    weights: LinearWeights,
    bias: Option<Array>,
}

enum LinearWeights {
    Dense(Array),
    QuantizedQ8 {
        weight: Array,
        scales: Array,
        biases: Array,
    },
}

impl Linear {
    fn load(
        weights: &mut HashMap<String, Array>,
        name: &str,
        format: WeightFormat,
    ) -> Result<Self, Exception> {
        let weight = take(weights, &format!("{name}.weight"))?;
        let linear_weights = match format {
            WeightFormat::Dense => {
                if weights.contains_key(&format!("{name}.scales"))
                    || weights.contains_key(&format!("{name}.biases"))
                    || !matches!(
                        weight.dtype(),
                        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
                    )
                {
                    return Err(Exception::custom(format!(
                        "{name} is not a dense floating-point linear layer"
                    )));
                }
                LinearWeights::Dense(weight)
            }
            WeightFormat::QuantizedQ8 => LinearWeights::QuantizedQ8 {
                weight,
                scales: take(weights, &format!("{name}.scales"))?,
                biases: take(weights, &format!("{name}.biases"))?,
            },
        };
        Ok(Self {
            weights: linear_weights,
            bias: weights.remove(&format!("{name}.bias")),
        })
    }

    fn forward(&self, input: &Array) -> Result<Array, Exception> {
        let output = match &self.weights {
            LinearWeights::Dense(weight) => input.matmul(&weight.transpose()?)?,
            LinearWeights::QuantizedQ8 {
                weight,
                scales,
                biases,
            } => quantized_matmul(input, weight, scales, Some(biases), true, 64, 8)?,
        };
        Ok(match &self.bias {
            Some(bias) => output + bias,
            None => output,
        })
    }
}

struct Conv1d {
    weight: Array,
    bias: Option<Array>,
    groups: i32,
}

impl Conv1d {
    fn load(
        weights: &mut HashMap<String, Array>,
        name: &str,
        groups: i32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            weight: take(weights, &format!("{name}.weight"))?,
            bias: weights.remove(&format!("{name}.bias")),
            groups,
        })
    }

    fn forward(&self, input: &Array, padding: i32) -> Result<Array, Exception> {
        let input = if padding == 0 {
            input.clone()
        } else {
            pad(input, &[(0, 0), (padding, padding), (0, 0)], None, None)?
        };
        let output = conv1d(input, &self.weight, 1, 0, 1, self.groups)?;
        Ok(match &self.bias {
            Some(bias) => output + bias,
            None => output,
        })
    }
}

struct LayerNorm {
    weight: Array,
    bias: Array,
}

impl LayerNorm {
    fn load(weights: &mut HashMap<String, Array>, name: &str) -> Result<Self, Exception> {
        Ok(Self {
            weight: take(weights, &format!("{name}.weight"))?,
            bias: take(weights, &format!("{name}.bias"))?,
        })
    }

    fn forward(&self, input: &Array) -> Result<Array, Exception> {
        layer_norm(input, Some(&self.weight), Some(&self.bias), 1.0e-5)
    }
}

struct BatchNorm {
    weight: Array,
    bias: Array,
    mean: Array,
    variance: Array,
}

impl BatchNorm {
    fn load(weights: &mut HashMap<String, Array>, name: &str) -> Result<Self, Exception> {
        Ok(Self {
            weight: take(weights, &format!("{name}.weight"))?,
            bias: take(weights, &format!("{name}.bias"))?,
            mean: take(weights, &format!("{name}.running_mean"))?,
            variance: take(weights, &format!("{name}.running_var"))?,
        })
    }

    fn forward(&self, input: &Array) -> Result<Array, Exception> {
        Ok((input - &self.mean) / (&self.variance + 1.0e-5f32).sqrt()? * &self.weight + &self.bias)
    }
}

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    position: Linear,
    output: Linear,
    bias_u: Array,
    bias_v: Array,
}

impl Attention {
    fn load(
        weights: &mut HashMap<String, Array>,
        prefix: &str,
        format: WeightFormat,
    ) -> Result<Self, Exception> {
        Ok(Self {
            query: Linear::load(weights, &format!("{prefix}.linear_q"), format)?,
            key: Linear::load(weights, &format!("{prefix}.linear_k"), format)?,
            value: Linear::load(weights, &format!("{prefix}.linear_v"), format)?,
            position: Linear::load(weights, &format!("{prefix}.linear_pos"), format)?,
            output: Linear::load(weights, &format!("{prefix}.linear_out"), format)?,
            bias_u: take(weights, &format!("{prefix}.pos_bias_u"))?,
            bias_v: take(weights, &format!("{prefix}.pos_bias_v"))?,
        })
    }

    fn forward(&self, input: &Array, position: &Array) -> Result<Array, Exception> {
        let [batch, length, _] = input.shape() else {
            return Err(Exception::custom(
                "Parakeet attention expects [batch, time, width]",
            ));
        };
        let query = self
            .query
            .forward(input)?
            .reshape(&[*batch, *length, HEADS, HEAD_DIM])?;
        let query_u = (&query + &self.bias_u).transpose_axes(&[0, 2, 1, 3])?;
        let query_v = (query + &self.bias_v).transpose_axes(&[0, 2, 1, 3])?;
        let key = self
            .key
            .forward(input)?
            .reshape(&[*batch, *length, HEADS, HEAD_DIM])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let value = self
            .value
            .forward(input)?
            .reshape(&[*batch, *length, HEADS, HEAD_DIM])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let position = self.position.forward(position)?;
        let position_length = position.dim(1);
        let position = position
            .reshape(&[1, position_length, HEADS, HEAD_DIM])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let relative = query_v.matmul(position.transpose_axes(&[0, 1, 3, 2])?)?;
        let relative = pad(relative, &[(0, 0), (0, 0), (0, 0), (1, 0)], None, None)?
            .reshape(&[*batch, HEADS, position_length + 1, *length])?
            .index((Ellipsis, 1.., ..))
            .reshape(&[*batch, HEADS, *length, position_length])?
            .index((Ellipsis, ..*length))
            * (HEAD_DIM as f32).powf(-0.5);
        let output = scaled_dot_product_attention(
            query_u,
            key,
            value,
            (HEAD_DIM as f32).powf(-0.5),
            Some(ScaledDotProductAttentionMask::Array(&relative)),
            None,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[*batch, *length, WIDTH])?;
        self.output.forward(&output)
    }
}

struct Block {
    norm_ff1: LayerNorm,
    ff1_in: Linear,
    ff1_out: Linear,
    norm_attn: LayerNorm,
    attention: Attention,
    norm_conv: LayerNorm,
    conv_pw1: Conv1d,
    conv_dw: Conv1d,
    conv_norm: BatchNorm,
    conv_pw2: Conv1d,
    norm_ff2: LayerNorm,
    ff2_in: Linear,
    ff2_out: Linear,
    norm_out: LayerNorm,
}

impl Block {
    fn load(
        weights: &mut HashMap<String, Array>,
        index: usize,
        format: WeightFormat,
    ) -> Result<Self, Exception> {
        let prefix = format!("encoder.layers.{index}");
        Ok(Self {
            norm_ff1: LayerNorm::load(weights, &format!("{prefix}.norm_feed_forward1"))?,
            ff1_in: Linear::load(weights, &format!("{prefix}.feed_forward1.linear1"), format)?,
            ff1_out: Linear::load(weights, &format!("{prefix}.feed_forward1.linear2"), format)?,
            norm_attn: LayerNorm::load(weights, &format!("{prefix}.norm_self_att"))?,
            attention: Attention::load(weights, &format!("{prefix}.self_attn"), format)?,
            norm_conv: LayerNorm::load(weights, &format!("{prefix}.norm_conv"))?,
            conv_pw1: Conv1d::load(weights, &format!("{prefix}.conv.pointwise_conv1"), 1)?,
            conv_dw: Conv1d::load(weights, &format!("{prefix}.conv.depthwise_conv"), WIDTH)?,
            conv_norm: BatchNorm::load(weights, &format!("{prefix}.conv.batch_norm"))?,
            conv_pw2: Conv1d::load(weights, &format!("{prefix}.conv.pointwise_conv2"), 1)?,
            norm_ff2: LayerNorm::load(weights, &format!("{prefix}.norm_feed_forward2"))?,
            ff2_in: Linear::load(weights, &format!("{prefix}.feed_forward2.linear1"), format)?,
            ff2_out: Linear::load(weights, &format!("{prefix}.feed_forward2.linear2"), format)?,
            norm_out: LayerNorm::load(weights, &format!("{prefix}.norm_out"))?,
        })
    }

    fn feed_forward(&self, input: &Array, first: bool) -> Result<Array, Exception> {
        let (inner, outer) = if first {
            (&self.ff1_in, &self.ff1_out)
        } else {
            (&self.ff2_in, &self.ff2_out)
        };
        let value = inner.forward(input)?;
        outer.forward(&(&value * sigmoid(&value)?))
    }

    fn convolution(&self, input: &Array) -> Result<Array, Exception> {
        let projected = self.conv_pw1.forward(input, 0)?;
        let halves = projected.split_equal(2, -1)?;
        let gated = &halves[0] * sigmoid(&halves[1])?;
        let value = self.conv_dw.forward(&gated, 4)?;
        let value = self.conv_norm.forward(&value)?;
        let value = &value * sigmoid(&value)?;
        self.conv_pw2.forward(&value, 0)
    }

    fn forward(&self, input: Array, position: &Array) -> Result<Array, Exception> {
        let ff1 = self.feed_forward(&self.norm_ff1.forward(&input)?, true)? * 0.5f32;
        let input = input + ff1;
        let attention = self
            .attention
            .forward(&self.norm_attn.forward(&input)?, position)?;
        let input = input + attention;
        let convolution = self.convolution(&self.norm_conv.forward(&input)?)?;
        let input = input + convolution;
        let ff2 = self.feed_forward(&self.norm_ff2.forward(&input)?, false)? * 0.5f32;
        self.norm_out.forward(&(input + ff2))
    }
}

struct Conv2d {
    weight: Array,
    bias: Array,
    kernel: i32,
    stride: i32,
    padding: i32,
    groups: i32,
}

impl Conv2d {
    fn load(
        weights: &mut HashMap<String, Array>,
        name: &str,
        kernel: i32,
        stride: i32,
        padding: i32,
        groups: i32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            weight: take(weights, &format!("{name}.weight"))?,
            bias: take(weights, &format!("{name}.bias"))?,
            kernel,
            stride,
            padding,
            groups,
        })
    }

    fn forward(&self, input: &Array) -> Result<Array, Exception> {
        let _ = self.kernel;
        Ok(conv2d(
            input,
            &self.weight,
            (self.stride, self.stride),
            (self.padding, self.padding),
            (1, 1),
            self.groups,
        )? + &self.bias)
    }
}

struct Subsampling {
    conv0: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    output: Linear,
}

impl Subsampling {
    fn load(weights: &mut HashMap<String, Array>, format: WeightFormat) -> Result<Self, Exception> {
        Ok(Self {
            conv0: Conv2d::load(weights, "encoder.pre_encode.conv.0", 3, 2, 1, 1)?,
            conv2: Conv2d::load(weights, "encoder.pre_encode.conv.2", 3, 2, 1, 256)?,
            conv3: Conv2d::load(weights, "encoder.pre_encode.conv.3", 1, 1, 0, 1)?,
            conv5: Conv2d::load(weights, "encoder.pre_encode.conv.5", 3, 2, 1, 256)?,
            conv6: Conv2d::load(weights, "encoder.pre_encode.conv.6", 1, 1, 0, 1)?,
            output: Linear::load(weights, "encoder.pre_encode.out", format)?,
        })
    }

    fn forward(&self, input: &Array) -> Result<Array, Exception> {
        let zero = Array::from_f32(0.0);
        let input = input.expand_dims(1)?.transpose_axes(&[0, 2, 3, 1])?;
        let value = maximum(self.conv0.forward(&input)?, &zero)?;
        let value = self.conv2.forward(&value)?;
        let value = maximum(self.conv3.forward(&value)?, &zero)?;
        let value = self.conv5.forward(&value)?;
        let value = maximum(self.conv6.forward(&value)?, &zero)?;
        let value = value
            .transpose_axes(&[0, 3, 1, 2])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let batch = value.dim(0);
        let length = value.dim(1);
        self.output.forward(&value.reshape(&[batch, length, -1])?)
    }
}

pub(super) struct Encoder {
    subsampling: Subsampling,
    blocks: Vec<Block>,
    projection: Linear,
    position: Array,
}

impl Encoder {
    pub(super) fn load(
        weights: &mut HashMap<String, Array>,
        format: WeightFormat,
    ) -> Result<Self, Exception> {
        let positions = arange::<f32, f32>(Some(4_999.0), -5_000.0, Some(-1.0))?.expand_dims(1)?;
        let exponent = arange::<f32, f32>(Some(0.0), WIDTH as f32, Some(2.0))?
            * (-(10000.0f32.ln()) / WIDTH as f32);
        let phase = positions * exponent.exp()?;
        let position = stack(&[phase.sin()?, phase.cos()?], -1)?
            .reshape(&[1, 9_999, WIDTH])?
            .as_type::<bf16>()?;
        position.eval()?;
        Ok(Self {
            subsampling: Subsampling::load(weights, format)?,
            blocks: (0..24)
                .map(|index| Block::load(weights, index, format))
                .collect::<Result<Vec<_>, _>>()?,
            projection: Linear::load(weights, "joint.enc", format)?,
            position,
        })
    }

    pub(super) fn forward(&self, mel: &Array) -> Result<Array, Exception> {
        let mut value = self.subsampling.forward(mel)? * (WIDTH as f32).sqrt();
        let length = value.dim(1);
        if length > 5_000 {
            return Err(Exception::custom(format!(
                "Parakeet encoder length {length} exceeds the 5000-frame position cache"
            )));
        }
        let start = 5_000 - length;
        let end = 5_000 + length - 1;
        let position = self.position.index((.., start..end, ..));
        for block in &self.blocks {
            value = block.forward(value, &position)?;
        }
        self.projection.forward(&value)
    }
}

fn take(weights: &mut HashMap<String, Array>, name: &str) -> Result<Array, Exception> {
    weights
        .remove(name)
        .ok_or_else(|| Exception::custom(format!("missing Parakeet MLX weight `{name}`")))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use mlx_rs::Array;

    use super::{Linear, WeightFormat};

    #[test]
    fn dense_linear_uses_unquantized_weights() {
        let mut weights = HashMap::from([(
            "test.weight".to_string(),
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
        )]);
        let linear = Linear::load(&mut weights, "test", WeightFormat::Dense).unwrap();
        let input = Array::from_slice(&[2.0f32, 3.0], &[1, 2]);
        assert_eq!(
            linear
                .forward(&input)
                .unwrap()
                .to_vec_cast::<f32>()
                .unwrap(),
            [8.0, 18.0]
        );
    }

    #[test]
    fn weight_mode_rejects_the_other_artifact_format() {
        let weight = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let mut dense = HashMap::from([("test.weight".to_string(), weight.clone())]);
        assert!(Linear::load(&mut dense, "test", WeightFormat::QuantizedQ8).is_err());

        let mut quantized = HashMap::from([
            ("test.weight".to_string(), weight),
            (
                "test.scales".to_string(),
                Array::from_slice(&[1.0f32], &[1]),
            ),
        ]);
        assert!(Linear::load(&mut quantized, "test", WeightFormat::Dense).is_err());
    }
}
