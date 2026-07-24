use ndarray::Array4;
use opencv::core::{Mat, Vec3b};
use opencv::prelude::MatTraitConst;

use crate::RameResult;
use crate::image::Image;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::PreprocessBackend;
use crate::preprocess::vision::VisionTensorOutput;

#[derive(Debug, Clone, Copy, Default)]
pub struct OpenCvVisionBackend;

#[doc(hidden)]
pub struct OpenCvVisionState {
    pub(super) image: Mat,
    pub(super) source_width: i32,
    pub(super) source_height: i32,
    pub(super) scale_factor: [f32; 2],
    pub(super) tensor: Option<Array4<f32>>,
}

impl PreprocessBackend for OpenCvVisionBackend {
    type Source = Image;
    type State = OpenCvVisionState;
    type Output = VisionTensorOutput;

    fn state(&self, image: &Self::Source) -> RameResult<Self::State> {
        OpenCvVisionState::new(image)
    }

    fn finish(&self, state: Self::State) -> RameResult<Self::Output> {
        let tensor = state.tensor.ok_or(PreprocessError::MissingOutput)?;

        Ok(VisionTensorOutput {
            tensor,
            scale_factor: state.scale_factor,
        })
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
            tensor: None,
        })
    }
}

fn rgb_pixels(data: &[u8]) -> Vec<Vec3b> {
    data.chunks_exact(3)
        .map(|pixel| Vec3b::from([pixel[0], pixel[1], pixel[2]]))
        .collect()
}
