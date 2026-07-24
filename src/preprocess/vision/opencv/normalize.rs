use ndarray::Array3;
use opencv::core::{Mat, MatTraitConstManual, Vec3b};
use opencv::prelude::MatTraitConst;

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::PreprocessOp;
use crate::preprocess::vision::opencv::state::{OpenCvVisionBackend, OpenCvVisionState};
use crate::preprocess::vision::{NormalizeImage, PixelScale};

impl PreprocessOp<OpenCvVisionBackend> for NormalizeImage {
    fn apply(&self, state: &mut OpenCvVisionState) -> RameResult<()> {
        state.normalized_image = Some(normalize_image(&state.image, *self)?);
        Ok(())
    }
}

fn normalize_image(image: &Mat, op: NormalizeImage) -> RameResult<Array3<f32>> {
    let size = image.size().map_err(PreprocessError::from)?;
    let width = size.width as usize;
    let height = size.height as usize;
    let image = image.data_typed::<Vec3b>().map_err(PreprocessError::from)?;
    let scale = pixel_scale_value(op.scale);
    let mut normalized = Array3::<f32>::zeros((height, width, 3));

    for y in 0..height {
        for x in 0..width {
            let pixel = image[y * width + x];
            for channel in 0..3 {
                normalized[[y, x, channel]] =
                    (pixel[channel] as f32 * scale - op.mean[channel]) / op.std[channel];
            }
        }
    }

    Ok(normalized)
}

fn pixel_scale_value(scale: PixelScale) -> f32 {
    match scale {
        PixelScale::OneOver255 => 1.0 / 255.0,
    }
}
