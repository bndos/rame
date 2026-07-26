use opencv::core::{Mat, Size};
use opencv::imgproc;

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::PreprocessOp;
use crate::preprocess::vision::opencv::state::{
    OpenCvVisionBackend, OpenCvVisionBatch, OpenCvVisionState,
};
use crate::preprocess::vision::{Interpolation, Resize, ResizeMode};

impl PreprocessOp<OpenCvVisionBackend> for Resize {
    fn apply(&self, batch: &mut OpenCvVisionBatch) -> RameResult<()> {
        for item in &mut batch.items {
            item.image = self.resize_mat(&item.image)?;
            item.scale_factor = self.scale_factor(item);
        }

        Ok(())
    }
}

impl Resize {
    fn resize_mat(&self, source: &Mat) -> RameResult<Mat> {
        let mut resized = Mat::default();
        match self.mode {
            ResizeMode::FixedSize { width, height } => {
                imgproc::resize(
                    source,
                    &mut resized,
                    Size::new(width as i32, height as i32),
                    0.0,
                    0.0,
                    self.opencv_interpolation(),
                )
                .map_err(PreprocessError::from)?;
            }
            ResizeMode::Scale {
                scale_width,
                scale_height,
            } => {
                imgproc::resize(
                    source,
                    &mut resized,
                    Size::default(),
                    scale_width as f64,
                    scale_height as f64,
                    self.opencv_interpolation(),
                )
                .map_err(PreprocessError::from)?;
            }
        }

        Ok(resized)
    }

    fn scale_factor(&self, state: &OpenCvVisionState) -> [f32; 2] {
        match self.mode {
            ResizeMode::FixedSize { width, height } => [
                height as f32 / state.source_height as f32,
                width as f32 / state.source_width as f32,
            ],
            ResizeMode::Scale {
                scale_width,
                scale_height,
            } => [scale_height, scale_width],
        }
    }

    fn opencv_interpolation(&self) -> i32 {
        match self.interpolation {
            Interpolation::Cubic => imgproc::INTER_CUBIC,
        }
    }
}
