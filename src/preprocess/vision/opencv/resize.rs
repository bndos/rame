use opencv::core::{Mat, Size};
use opencv::imgproc;

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::PreprocessOp;
use crate::preprocess::vision::opencv::state::{OpenCvVisionBackend, OpenCvVisionState};
use crate::preprocess::vision::{Interpolation, Resize, ResizeMode};

impl PreprocessOp<OpenCvVisionBackend> for Resize {
    fn apply(&self, state: &mut OpenCvVisionState) -> RameResult<()> {
        state.image = resize_image(&state.image, *self)?;
        state.scale_factor = scale_factor(*self, state);
        Ok(())
    }
}

fn resize_image(source: &Mat, resize: Resize) -> RameResult<Mat> {
    let mut resized = Mat::default();
    match resize.mode {
        ResizeMode::FixedSize { width, height } => {
            imgproc::resize(
                source,
                &mut resized,
                Size::new(width as i32, height as i32),
                0.0,
                0.0,
                interpolation_to_opencv(resize.interpolation),
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
                interpolation_to_opencv(resize.interpolation),
            )
            .map_err(PreprocessError::from)?;
        }
    }

    Ok(resized)
}

fn scale_factor(resize: Resize, state: &OpenCvVisionState) -> [f32; 2] {
    match resize.mode {
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

fn interpolation_to_opencv(interpolation: Interpolation) -> i32 {
    match interpolation {
        Interpolation::Cubic => imgproc::INTER_CUBIC,
    }
}
