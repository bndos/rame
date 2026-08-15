use rayon::prelude::*;

use super::cpu;
use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp};
use crate::preprocess::vision::opencv::OpenCvVisionBackend;
use crate::preprocess::vision::opencv::state::{OpenCvImage, OpenCvVisionData, OpenCvVisionState};
use crate::preprocess::vision::{Resize, ResizeMode};

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
                item.image = self.resize_image(&item.image, &batch.device)?;
                item.scale_factor = self.scale_factor(item);
                Ok(())
            })?;

        Ok(OpenCvVisionData::ImageBatch(batch))
    }
}

impl Resize {
    fn resize_image(
        &self,
        source: &OpenCvImage<'_>,
        device: &candle_core::Device,
    ) -> RameResult<OpenCvImage<'static>> {
        match device {
            candle_core::Device::Cpu => cpu::resize(self, source).map(OpenCvImage::Owned),
            candle_core::Device::Cuda(_) => Err(PreprocessError::UnsupportedBackendOp {
                backend: "OpenCV CUDA",
                op: "Resize",
            }
            .into()),
            candle_core::Device::Metal(_) => Err(PreprocessError::UnsupportedBackendOp {
                backend: "OpenCV Metal",
                op: "Resize",
            }
            .into()),
        }
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
}
