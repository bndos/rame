use rayon::prelude::*;

use super::cpu;
use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp};
use crate::preprocess::vision::opencv::OpenCvVisionBackend;
use crate::preprocess::vision::opencv::state::{OpenCvImage, OpenCvVisionData};
use crate::preprocess::vision::{Resize, ResizeMode};

impl PreprocessOp<OpenCvVisionBackend> for Resize {
    fn forward<'a>(
        &self,
        data: <OpenCvVisionBackend as PreprocessBackend>::Data<'a>,
    ) -> RameResult<<OpenCvVisionBackend as PreprocessBackend>::Data<'a>> {
        let mut batch = data.into_image_batch()?;

        let device = &batch.device;
        let source_sizes = &batch.source_sizes;
        batch
            .images
            .par_iter_mut()
            .zip(batch.scale_factors.par_iter_mut())
            .enumerate()
            .try_for_each(|(index, (image, scale_factor))| -> RameResult<()> {
                *image = self.resize_image(image, device)?;
                *scale_factor = self.scale_factor(source_sizes[index]);
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

    fn scale_factor(&self, source_size: [i32; 2]) -> [f32; 2] {
        let [source_height, source_width] = source_size;
        match self.mode {
            ResizeMode::FixedSize { width, height } => [
                height as f32 / source_height as f32,
                width as f32 / source_width as f32,
            ],
            ResizeMode::Scale {
                scale_width,
                scale_height,
            } => [scale_height, scale_width],
        }
    }
}
