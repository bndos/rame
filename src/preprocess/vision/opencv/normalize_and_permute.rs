use ndarray::Array4;
use opencv::core::{Mat, MatTraitConstManual, Vec3b};
use opencv::prelude::MatTraitConst;

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::PreprocessOp;
use crate::preprocess::vision::opencv::state::{OpenCvVisionBackend, OpenCvVisionState};
use crate::preprocess::vision::{NormalizeAndPermute, PixelScale, TensorLayout};

impl PreprocessOp<OpenCvVisionBackend> for NormalizeAndPermute {
    fn apply(&self, state: &mut OpenCvVisionState) -> RameResult<()> {
        let scale = pixel_scale_value(self.scale);
        state.tensor = Some(mat_to_f32_tensor(&state.image, self.layout, scale)?);
        Ok(())
    }
}

fn mat_to_f32_tensor(image: &Mat, layout: TensorLayout, scale: f32) -> RameResult<Array4<f32>> {
    let size = image.size().map_err(PreprocessError::from)?;
    let width = size.width as usize;
    let height = size.height as usize;
    let image = image.data_typed::<Vec3b>().map_err(PreprocessError::from)?;
    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    match layout {
        TensorLayout::Nchw => {
            for y in 0..height {
                for x in 0..width {
                    let pixel = image[y * width + x];
                    for channel in 0..3 {
                        tensor[[0, channel, y, x]] = pixel[channel] as f32 * scale;
                    }
                }
            }
        }
    }

    Ok(tensor)
}

fn pixel_scale_value(scale: PixelScale) -> f32 {
    match scale {
        PixelScale::OneOver255 => 1.0 / 255.0,
    }
}
