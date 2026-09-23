use crate::RameResult;
use crate::models::ModelError;
use crate::tensor::{Tensor, TensorError};

#[derive(Debug, Clone)]
pub struct TdtJointOutput {
    token_logits: Tensor,
    duration_logits: Tensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdtJointLayout {
    num_token_classes: usize,
    num_duration_classes: usize,
}

impl TdtJointLayout {
    pub const fn new(num_token_classes: usize, num_duration_classes: usize) -> Self {
        Self {
            num_token_classes,
            num_duration_classes,
        }
    }

    pub const fn num_token_classes(self) -> usize {
        self.num_token_classes
    }

    pub const fn num_duration_classes(self) -> usize {
        self.num_duration_classes
    }

    pub const fn joint_logit_count(self) -> usize {
        self.num_token_classes + self.num_duration_classes
    }
}

impl From<&super::TdtDecodingConfig> for TdtJointLayout {
    fn from(config: &super::TdtDecodingConfig) -> Self {
        Self::new(config.num_token_classes(), config.durations().len())
    }
}

impl TdtJointOutput {
    pub fn from_combined(logits: Tensor, layout: TdtJointLayout) -> RameResult<Self> {
        let dimensions = logits.dims();
        let Some((&width, leading_dimensions)) = dimensions.split_last() else {
            return Err(ModelError::InvalidTensorShape {
                name: "TDT joint logits".into(),
                expected: format!(
                    "rank >= 1 with last dimension {}",
                    layout.joint_logit_count()
                ),
                actual: dimensions.to_vec(),
            }
            .into());
        };
        if width != layout.joint_logit_count() {
            return Err(ModelError::InvalidTensorShape {
                name: "TDT joint logits".into(),
                expected: format!("last dimension {}", layout.joint_logit_count()),
                actual: dimensions.to_vec(),
            }
            .into());
        }

        let output_axis = leading_dimensions.len();
        let token_logits = logits
            .narrow(output_axis, 0, layout.num_token_classes)
            .map_err(TensorError::from)?;
        let duration_logits = logits
            .narrow(
                output_axis,
                layout.num_token_classes,
                layout.num_duration_classes,
            )
            .map_err(TensorError::from)?;

        Ok(Self {
            token_logits,
            duration_logits,
        })
    }

    pub fn token_logits(&self) -> &Tensor {
        &self.token_logits
    }

    pub fn duration_logits(&self) -> &Tensor {
        &self.duration_logits
    }

    pub fn into_parts(self) -> (Tensor, Tensor) {
        (self.token_logits, self.duration_logits)
    }
}

#[cfg(test)]
mod tests {
    use crate::RameError;
    use crate::models::ModelError;
    use crate::runtime::transducer::tdt::TdtDecodingConfig;
    use crate::tensor::Tensor;

    use super::{TdtJointLayout, TdtJointOutput};

    #[test]
    fn splits_combined_joint_logits_without_changing_leading_dimensions() {
        let layout =
            TdtJointLayout::from(&TdtDecodingConfig::new(8_193, 8_192, [0, 1, 2, 3, 4], 10));
        let joint_logit_count = layout.joint_logit_count();
        let logits = Tensor::from_vec(
            vec![0.0f32; 2 * joint_logit_count],
            (2, joint_logit_count),
            &crate::tensor::Device::Cpu,
        )
        .unwrap();

        let output = TdtJointOutput::from_combined(logits, layout).unwrap();

        assert_eq!(output.token_logits().dims(), &[2, 8_193]);
        assert_eq!(output.duration_logits().dims(), &[2, 5]);
    }

    #[test]
    fn rejects_an_incompatible_joint_width() {
        let layout =
            TdtJointLayout::from(&TdtDecodingConfig::new(8_193, 8_192, [0, 1, 2, 3, 4], 10));
        let logits =
            Tensor::from_vec(vec![0.0f32; 8_197], 8_197, &crate::tensor::Device::Cpu).unwrap();

        let error = TdtJointOutput::from_combined(logits, layout).unwrap_err();

        assert!(matches!(
            error,
            RameError::Model(ModelError::InvalidTensorShape { .. })
        ));
    }
}
