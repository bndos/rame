use opencv::core::{Mat, Size, ToInputArray};
use opencv::imgproc;
use rayon::prelude::*;

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp};
use crate::preprocess::vision::opencv::OpenCvVisionBackend;
use crate::preprocess::vision::opencv::state::{OpenCvImage, OpenCvVisionData, OpenCvVisionState};
use crate::preprocess::vision::{Interpolation, Resize, ResizeMode};

impl PreprocessOp<OpenCvVisionBackend> for Resize {
    fn forward<'a>(
        &self,
        data: <OpenCvVisionBackend as PreprocessBackend>::Data<'a>,
    ) -> RameResult<<OpenCvVisionBackend as PreprocessBackend>::Data<'a>> {
        let mut batch = data.into_image_batch()?;

        batch
            .items
            .par_iter_mut()
            .try_for_each(|item| -> RameResult<()> {
                item.image = OpenCvImage::Owned(self.resize_mat(&item.image)?);
                item.scale_factor = self.scale_factor(item);
                Ok(())
            })?;

        Ok(OpenCvVisionData::ImageBatch(batch))
    }
}

impl Resize {
    fn resize_mat(&self, source: &OpenCvImage<'_>) -> RameResult<Mat> {
        match source {
            OpenCvImage::Borrowed(source) => self.resize_source(source),
            OpenCvImage::Owned(source) => self.resize_source(source),
        }
    }

    fn resize_source(&self, source: &impl ToInputArray) -> RameResult<Mat> {
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

    fn scale_factor(&self, state: &OpenCvVisionState<'_>) -> [f32; 2] {
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
