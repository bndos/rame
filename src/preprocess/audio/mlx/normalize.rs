use half::bf16;
use mlx_rs::ops::std_axis;

use super::state::{MlxAudioBackend, Stage};
use crate::RameResult;
use crate::preprocess::audio::NormalizeFeatures;
use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp};
use crate::tensor::DType;

impl PreprocessOp<MlxAudioBackend> for NormalizeFeatures {
    fn forward<'a>(
        &self,
        mut data: <MlxAudioBackend as PreprocessBackend>::Data<'a>,
    ) -> RameResult<<MlxAudioBackend as PreprocessBackend>::Data<'a>> {
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(super::invalid(
                "feature normalization epsilon must be positive",
            ));
        }
        let Stage::LogMel(features) = data.stage else {
            return Err(super::invalid(
                "feature normalization expects log-mel input",
            ));
        };
        let mut output = Vec::with_capacity(features.len());
        for mel in features {
            let mean = mel.mean_axis(1, true).map_err(super::mlx_error)?;
            let std = std_axis(&mel, 1, true, 0).map_err(super::mlx_error)?;
            let normalized = ((mel - mean) / (std + self.epsilon))
                .transpose_axes(&[1, 0])
                .map_err(super::mlx_error)?
                .expand_dims(0)
                .map_err(super::mlx_error)?;
            let normalized = match self.output_dtype {
                DType::BF16 => normalized.as_type::<bf16>().map_err(super::mlx_error)?,
                DType::F32 => normalized,
                _ => return Err(super::invalid("unsupported MLX audio feature dtype")),
            };
            output.push(normalized);
        }
        data.stage = Stage::Output(output);
        Ok(data)
    }
}
