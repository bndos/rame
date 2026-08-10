use ndarray::Array3;
use opencv::core::{MatTraitConstManual, Vec3b};
use opencv::prelude::MatTraitConst;

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::vision::NormalizeImage;
use crate::preprocess::vision::opencv::state::{OpenCvImage, OpenCvVisionBatch};

impl NormalizeImage {
    pub(super) fn apply_opencv(&self, batch: &mut OpenCvVisionBatch<'_>) -> RameResult<()> {
        batch.normalized_images = Some(
            batch
                .items
                .iter()
                .map(|item| self.normalize_image(&item.image))
                .collect::<RameResult<Vec<_>>>()?,
        );

        Ok(())
    }
}

impl NormalizeImage {
    fn normalize_image(&self, image: &OpenCvImage<'_>) -> RameResult<Array3<f32>> {
        match image {
            OpenCvImage::Borrowed(image) => self.normalize_mat(image),
            OpenCvImage::Owned(image) => self.normalize_mat(image),
        }
    }

    fn normalize_mat<T>(&self, image: &T) -> RameResult<Array3<f32>>
    where
        T: MatTraitConst + MatTraitConstManual,
    {
        let size = image.size().map_err(PreprocessError::from)?;
        let width = size.width as usize;
        let height = size.height as usize;
        let image = image.data_typed::<Vec3b>().map_err(PreprocessError::from)?;
        let coefficients = self.coefficients();
        let mut normalized = Array3::<f32>::zeros((height, width, 3));

        for y in 0..height {
            for x in 0..width {
                let pixel = image[y * width + x];
                normalized[[y, x, 0]] =
                    pixel[0] as f32 * coefficients.scale[0] + coefficients.bias[0];
                normalized[[y, x, 1]] =
                    pixel[1] as f32 * coefficients.scale[1] + coefficients.bias[1];
                normalized[[y, x, 2]] =
                    pixel[2] as f32 * coefficients.scale[2] + coefficients.bias[2];
            }
        }

        Ok(normalized)
    }
}

impl NormalizeImage {
    pub(super) fn coefficients(self) -> NormalizationCoefficients {
        NormalizationCoefficients {
            scale: [
                self.scale / self.std[0],
                self.scale / self.std[1],
                self.scale / self.std[2],
            ],
            bias: [
                -self.mean[0] / self.std[0],
                -self.mean[1] / self.std[1],
                -self.mean[2] / self.std[2],
            ],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct NormalizationCoefficients {
    pub(super) scale: [f32; 3],
    pub(super) bias: [f32; 3],
}
