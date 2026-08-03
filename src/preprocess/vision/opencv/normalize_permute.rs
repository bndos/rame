use ndarray::Array4;
use opencv::core::{self, Mat, Size, Vector};
use opencv::prelude::MatTraitConst;

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::vision::opencv::state::OpenCvVisionBatch;
use crate::preprocess::vision::{NormalizeImage, Permute, TensorLayout};

#[derive(Debug, Clone, Copy)]
pub(super) struct NormalizeAndPermute {
    normalize: NormalizeImage,
    permute: Permute,
}

impl NormalizeAndPermute {
    pub(super) fn new(normalize: NormalizeImage, permute: Permute) -> Self {
        Self { normalize, permute }
    }
}

impl NormalizeAndPermute {
    pub(super) fn apply_opencv(&self, batch: &mut OpenCvVisionBatch) -> RameResult<()> {
        match self.permute.layout {
            TensorLayout::Nchw => self.apply_nchw(batch),
        }
    }
}

impl NormalizeAndPermute {
    fn apply_nchw(&self, batch: &mut OpenCvVisionBatch) -> RameResult<()> {
        if batch.items.is_empty() {
            return Ok(());
        }

        let size = batch.items[0].image.size().map_err(PreprocessError::from)?;
        let (height, width) = (size.height as usize, size.width as usize);
        let mut tensor = Array4::zeros((batch.items.len(), 3, height, width));
        // Writeable slice of the tensor used for OpenCV ops (&mut [f32])
        let output = tensor
            .as_slice_memory_order_mut()
            .ok_or(PreprocessError::MissingOutput)?;
        // The number of pixels in a single channel
        let plane = height * width;

        for (index, item) in batch.items.iter_mut().enumerate() {
            ensure_size(&item.image, size)?;
            let start = index * 3 * plane;
            let end = start + 3 * plane;
            self.normalize_mat_into_nchw(
                &item.image,
                height,
                width,
                plane,
                &mut output[start..end],
            )?;
        }

        batch.tensor = Some(tensor);

        Ok(())
    }
}

impl NormalizeAndPermute {
    fn normalize_mat_into_nchw(
        &self,
        image: &Mat,
        height: usize,
        width: usize,
        plane: usize,
        output: &mut [f32],
    ) -> RameResult<()> {
        let mut channels = Vector::<Mat>::new();
        let coefficients = self.normalize.coefficients();
        core::split(image, &mut channels).map_err(PreprocessError::from)?;

        for channel in 0..3 {
            let source = channels.get(channel).map_err(PreprocessError::from)?;
            let start = channel * plane;
            let end = start + plane;
            // OpenCV view over `output`.
            let mut target = Mat::new_rows_cols_with_data_mut(
                height as i32,
                width as i32,
                &mut output[start..end],
            )
            .map_err(PreprocessError::from)?;

            source
                .convert_to(
                    &mut target,
                    core::CV_32FC1,
                    coefficients.scale[channel] as f64,
                    coefficients.bias[channel] as f64,
                )
                .map_err(PreprocessError::from)?;
        }

        Ok(())
    }
}

fn ensure_size(image: &Mat, expected: Size) -> RameResult<()> {
    let actual = image.size().map_err(PreprocessError::from)?;
    if actual == expected {
        return Ok(());
    }

    Err(PreprocessError::InvalidTensorShape {
        name: "image",
        expected: format!("[height={}, width={}]", expected.height, expected.width),
        actual: vec![actual.height as usize, actual.width as usize],
    }
    .into())
}

#[cfg(test)]
mod tests {
    use crate::image::Image;
    use crate::preprocess::vision::opencv::normalize_permute::NormalizeAndPermute;
    use crate::preprocess::vision::opencv::state::OpenCvVisionBatch;
    use crate::preprocess::vision::{NormalizeImage, Permute};

    #[test]
    fn matches_separate_normalize_and_permute_ops() {
        let images = vec![
            Image::from_rgb8(2, 1, vec![255, 0, 64, 32, 128, 255]).unwrap(),
            Image::from_rgb8(2, 1, vec![0, 255, 128, 255, 64, 32]).unwrap(),
        ];

        let normalize = NormalizeImage::new(0.5, [0.1, 0.2, 0.3], [1.0, 2.0, 4.0]);
        let permute = Permute::nchw();
        let mut unfused = OpenCvVisionBatch::new(&images).unwrap();
        normalize.apply_opencv(&mut unfused).unwrap();
        permute.apply_opencv(&mut unfused).unwrap();
        let unfused = unfused.finish().unwrap();

        let mut fused = OpenCvVisionBatch::new(&images).unwrap();
        NormalizeAndPermute::new(normalize, permute)
            .apply_opencv(&mut fused)
            .unwrap();
        let fused = fused.finish().unwrap();

        assert_eq!(fused.tensor.shape(), unfused.tensor.shape());
        assert_eq!(fused.image_shapes, unfused.image_shapes);
        assert_eq!(fused.scale_factors, unfused.scale_factors);

        assert_tensor_close(&fused.tensor, &unfused.tensor);
    }

    fn assert_tensor_close(left: &ndarray::Array4<f32>, right: &ndarray::Array4<f32>) {
        for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
            let diff = (left - right).abs();
            assert!(
                diff < 1e-5,
                "tensor mismatch at {index}: left={left}, right={right}, diff={diff}"
            );
        }
    }
}
