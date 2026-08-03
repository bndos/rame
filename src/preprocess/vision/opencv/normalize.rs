use ndarray::Array3;
use opencv::core::{Mat, MatTraitConstManual, Vec3b};
use opencv::prelude::MatTraitConst;

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::vision::NormalizeImage;
use crate::preprocess::vision::opencv::state::OpenCvVisionBatch;

impl NormalizeImage {
    pub(super) fn apply_opencv(&self, batch: &mut OpenCvVisionBatch) -> RameResult<()> {
        batch.normalized_images = Some(
            batch
                .items
                .iter()
                .map(|item| self.normalize_mat(&item.image))
                .collect::<RameResult<Vec<_>>>()?,
        );

        Ok(())
    }
}

impl NormalizeImage {
    fn normalize_mat(&self, image: &Mat) -> RameResult<Array3<f32>> {
        let size = image.size().map_err(PreprocessError::from)?;
        let width = size.width as usize;
        let height = size.height as usize;
        let image = image.data_typed::<Vec3b>().map_err(PreprocessError::from)?;
        let mut normalized = Array3::<f32>::zeros((height, width, 3));

        for y in 0..height {
            for x in 0..width {
                let pixel = image[y * width + x];
                for channel in 0..3 {
                    normalized[[y, x, channel]] = (pixel[channel] as f32 * self.scale
                        - self.mean[channel])
                        / self.std[channel];
                }
            }
        }

        Ok(normalized)
    }
}
