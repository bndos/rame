use ndarray::{Array2, Array3, Array4, Axis, s};
use opencv::core::{Mat, Vec3b};
use opencv::prelude::MatTraitConst;

use crate::RameResult;
use crate::image::Image;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::PreprocessBackend;
use crate::preprocess::vision::VisionBatchOutput;

#[derive(Debug, Clone, Copy, Default)]
pub struct OpenCvVisionBackend;

#[doc(hidden)]
pub struct OpenCvVisionState {
    pub(super) image: Mat,
    pub(super) source_width: i32,
    pub(super) source_height: i32,
    pub(super) scale_factor: [f32; 2],
    pub(super) normalized_image: Option<Array3<f32>>,
    pub(super) tensor: Option<Array4<f32>>,
}

#[doc(hidden)]
pub struct OpenCvVisionBatch {
    pub(super) items: Vec<OpenCvVisionState>,
}

impl PreprocessBackend for OpenCvVisionBackend {
    type Source = Image;
    type Batch = OpenCvVisionBatch;
    type Output = VisionBatchOutput;

    fn batch(&self, images: &[Self::Source]) -> RameResult<Self::Batch> {
        OpenCvVisionBatch::new(images)
    }

    fn finish(&self, batch: Self::Batch) -> RameResult<Self::Output> {
        batch.finish()
    }
}

impl OpenCvVisionState {
    fn new(image: &Image) -> RameResult<Self> {
        let source_size = image.size();
        let source_height = source_size.height as i32;
        let source_width = source_size.width as i32;

        let pixels = rgb_pixels(image.data());
        let image = Mat::new_rows_cols_with_data(source_height, source_width, &pixels)
            .map_err(PreprocessError::from)?
            .try_clone()
            .map_err(PreprocessError::from)?;

        Ok(Self {
            image,
            source_width,
            source_height,
            scale_factor: [1.0, 1.0],
            normalized_image: None,
            tensor: None,
        })
    }
}

impl OpenCvVisionBatch {
    fn new(images: &[Image]) -> RameResult<Self> {
        let items = images
            .iter()
            .map(OpenCvVisionState::new)
            .collect::<RameResult<Vec<_>>>()?;

        Ok(Self { items })
    }

    fn finish(self) -> RameResult<VisionBatchOutput> {
        let len = self.items.len();

        let mut batch_tensor: Option<Array4<f32>> = None;
        let mut image_shapes = Array2::zeros((len, 2));
        let mut scale_factors = Array2::zeros((len, 2));

        for (index, state) in self.items.into_iter().enumerate() {
            let tensor = state.tensor.ok_or(PreprocessError::MissingOutput)?;
            // Allocate the final [N, C, H, W] tensor on the first item, then copy each
            // item into its batch slot instead of collecting all item tensors and stacking.
            ensure_batch_tensor(&mut batch_tensor, len, &tensor)?;
            let shape = tensor.shape();
            let image_height = shape[2] as f32;
            let image_width = shape[3] as f32;
            let batch = batch_tensor
                .as_mut()
                .ok_or(PreprocessError::MissingOutput)?;

            batch
                .slice_mut(s![index, .., .., ..])
                .assign(&tensor.index_axis(Axis(0), 0));

            image_shapes[[index, 0]] = image_height;
            image_shapes[[index, 1]] = image_width;
            scale_factors[[index, 0]] = state.scale_factor[0];
            scale_factors[[index, 1]] = state.scale_factor[1];
        }

        Ok(VisionBatchOutput {
            tensor: batch_tensor.ok_or(PreprocessError::MissingOutput)?,
            image_shapes,
            scale_factors,
        })
    }
}

fn rgb_pixels(data: &[u8]) -> Vec<Vec3b> {
    data.chunks_exact(3)
        .map(|pixel| Vec3b::from([pixel[0], pixel[1], pixel[2]]))
        .collect()
}

fn ensure_batch_tensor(
    batch_tensor: &mut Option<Array4<f32>>,
    len: usize,
    tensor: &Array4<f32>,
) -> RameResult<()> {
    let shape = tensor.shape();

    if shape.len() != 4 {
        return Err(PreprocessError::InvalidTensorShape {
            name: "image",
            expected: "[1, channels, height, width]".to_string(),
            actual: shape.to_vec(),
        }
        .into());
    }

    if let Some(batch) = batch_tensor {
        let expected_shape = [1, batch.shape()[1], batch.shape()[2], batch.shape()[3]];
        if shape != expected_shape {
            return Err(PreprocessError::InvalidTensorShape {
                name: "image",
                expected: format!("{expected_shape:?}"),
                actual: shape.to_vec(),
            }
            .into());
        }
        return Ok(());
    }

    if shape[0] != 1 {
        return Err(PreprocessError::InvalidTensorShape {
            name: "image",
            expected: "[1, channels, height, width]".to_string(),
            actual: shape.to_vec(),
        }
        .into());
    }

    *batch_tensor = Some(Array4::zeros((len, shape[1], shape[2], shape[3])));
    Ok(())
}
