use ndarray::{Array2, Array3, Array4};
use opencv::core::Mat;
use opencv::prelude::MatTraitConst;

use crate::RameResult;
use crate::image::Image;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp};
use crate::preprocess::vision::opencv::normalize_permute::NormalizeAndPermute;
use crate::preprocess::vision::{VisionBatchOutput, VisionOp};

#[derive(Debug, Clone, Copy, Default)]
pub struct OpenCvVisionBackend;

#[doc(hidden)]
pub struct OpenCvVisionState {
    pub(super) image: Mat,
    pub(super) source_width: i32,
    pub(super) source_height: i32,
    pub(super) scale_factor: [f32; 2],
}

#[doc(hidden)]
pub struct OpenCvVisionBatch {
    pub(super) items: Vec<OpenCvVisionState>,
    pub(super) normalized_images: Option<Vec<Array3<f32>>>,
    pub(super) tensor: Option<Array4<f32>>,
}

impl PreprocessBackend for OpenCvVisionBackend {
    type Source = Image;
    type Batch = OpenCvVisionBatch;
    type Output = VisionBatchOutput;
    type Op = VisionOp;

    fn compile(&self, ops: &mut Vec<Self::Op>) {
        super::compile::compile(ops);
    }

    fn batch(&self, images: &[Self::Source]) -> RameResult<Self::Batch> {
        OpenCvVisionBatch::new(images)
    }

    fn finish(&self, batch: Self::Batch) -> RameResult<Self::Output> {
        batch.finish()
    }
}

impl PreprocessOp<OpenCvVisionBackend> for VisionOp {
    fn apply(&self, batch: &mut OpenCvVisionBatch) -> RameResult<()> {
        match *self {
            Self::Resize(op) => op.apply_opencv(batch),
            Self::NormalizeImage(op) => op.apply_opencv(batch),
            Self::Permute(op) => op.apply_opencv(batch),
            Self::NormalizeAndPermute { normalize, permute } => {
                NormalizeAndPermute::new(normalize, permute).apply_opencv(batch)
            }
        }
    }
}

impl OpenCvVisionState {
    fn new(image: &Image) -> RameResult<Self> {
        let source_size = image.size();
        let source_height = source_size.height as i32;
        let source_width = source_size.width as i32;

        let image = Mat::new_rows_cols_with_data(1, source_height * source_width * 3, image.data())
            .map_err(PreprocessError::from)?
            .reshape(3, source_height)
            .map_err(PreprocessError::from)?
            .try_clone()
            .map_err(PreprocessError::from)?;

        Ok(Self {
            image,
            source_width,
            source_height,
            scale_factor: [1.0, 1.0],
        })
    }
}

impl OpenCvVisionBatch {
    pub(super) fn new(images: &[Image]) -> RameResult<Self> {
        let items = images
            .iter()
            .map(OpenCvVisionState::new)
            .collect::<RameResult<Vec<_>>>()?;

        Ok(Self {
            items,
            normalized_images: None,
            tensor: None,
        })
    }

    pub(super) fn finish(self) -> RameResult<VisionBatchOutput> {
        let len = self.items.len();
        let mut image_shapes = Array2::zeros((len, 2));
        let mut scale_factors = Array2::zeros((len, 2));
        let tensor = self.tensor.ok_or(PreprocessError::MissingOutput)?;
        let shape = tensor.shape();

        for (index, state) in self.items.iter().enumerate() {
            image_shapes[[index, 0]] = shape[2] as f32;
            image_shapes[[index, 1]] = shape[3] as f32;
            scale_factors[[index, 0]] = state.scale_factor[0];
            scale_factors[[index, 1]] = state.scale_factor[1];
        }

        Ok(VisionBatchOutput {
            tensor,
            image_shapes,
            scale_factors,
        })
    }
}
