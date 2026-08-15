use opencv::core::{Mat, Size, ToInputArray};
use opencv::imgproc;

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::vision::opencv::state::OpenCvImage;
use crate::preprocess::vision::{Interpolation, Resize, ResizeMode};

pub(in crate::preprocess::vision::opencv) fn resize(
    op: &Resize,
    source: &OpenCvImage<'_>,
) -> RameResult<Mat> {
    match source {
        OpenCvImage::Borrowed(source) => resize_source(op, source),
        OpenCvImage::Owned(source) => resize_source(op, source),
    }
}

fn resize_source(op: &Resize, source: &impl ToInputArray) -> RameResult<Mat> {
    let mut resized = Mat::default();
    match op.mode {
        ResizeMode::FixedSize { width, height } => {
            imgproc::resize(
                source,
                &mut resized,
                Size::new(width as i32, height as i32),
                0.0,
                0.0,
                opencv_interpolation(op),
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
                opencv_interpolation(op),
            )
            .map_err(PreprocessError::from)?;
        }
    }

    Ok(resized)
}

fn opencv_interpolation(op: &Resize) -> i32 {
    match op.interpolation {
        Interpolation::Cubic => imgproc::INTER_CUBIC,
    }
}
