use opencv::core::{self, Mat, Size, ToInputArray, Vector};
use opencv::prelude::MatTraitConst;

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp};
use crate::preprocess::vision::opencv::OpenCvVisionBackend;
use crate::preprocess::vision::opencv::state::{OpenCvImage, OpenCvVisionBatch, OpenCvVisionData};
use crate::preprocess::vision::{NormalizeImage, TensorLayout, ToTensor, VisionBatchOutput};
use crate::tensor::Tensor;

impl PreprocessOp<OpenCvVisionBackend> for ToTensor {
    fn forward<'a>(
        &self,
        data: <OpenCvVisionBackend as PreprocessBackend>::Data<'a>,
    ) -> RameResult<<OpenCvVisionBackend as PreprocessBackend>::Data<'a>> {
        let batch = data.into_image_batch()?;
        let output = match self.layout {
            TensorLayout::Nchw => self.apply_nchw(batch)?,
        };

        Ok(OpenCvVisionData::TensorBatch(output))
    }
}

impl ToTensor {
    fn output(
        &self,
        batch: OpenCvVisionBatch<'_>,
        data: Vec<f32>,
        shape: [usize; 4],
    ) -> RameResult<VisionBatchOutput> {
        let len = batch.items.len();

        let tensor = Tensor::from_vec(data, &shape).map_err(|err| PreprocessError::Backend {
            backend: "candle",
            message: err.to_string(),
        })?;

        let mut image_shapes = vec![0.0; len * 2];
        let mut scale_factors = vec![0.0; len * 2];
        for (index, state) in batch.items.iter().enumerate() {
            let offset = index * 2;
            image_shapes[offset] = shape[2] as f32;
            image_shapes[offset + 1] = shape[3] as f32;
            scale_factors[offset] = state.scale_factor[0];
            scale_factors[offset + 1] = state.scale_factor[1];
        }

        let image_shapes = metadata_tensor(image_shapes, len)?;
        let scale_factors = metadata_tensor(scale_factors, len)?;

        Ok(VisionBatchOutput {
            tensor,
            image_shapes,
            scale_factors,
        })
    }

    fn apply_nchw(&self, batch: OpenCvVisionBatch<'_>) -> RameResult<VisionBatchOutput> {
        if batch.items.is_empty() {
            return self.output(batch, Vec::new(), [0, 3, 0, 0]);
        }

        let size = batch.items[0].image.size().map_err(PreprocessError::from)?;
        let (height, width) = (size.height as usize, size.width as usize);
        let len = batch.items.len();
        let mut tensor = vec![0.0; len * 3 * height * width];
        // The number of pixels in a single channel
        let plane = height * width;

        for (output, item) in tensor.chunks_mut(3 * plane).zip(batch.items.iter()) {
            ensure_size(&item.image, size)?;
            self.image_into_nchw(&item.image, height, width, plane, output)?;
        }

        self.output(batch, tensor, [len, 3, height, width])
    }

    fn image_into_nchw(
        &self,
        image: &OpenCvImage<'_>,
        height: usize,
        width: usize,
        plane: usize,
        output: &mut [f32],
    ) -> RameResult<()> {
        match image {
            OpenCvImage::Borrowed(image) => self.mat_into_nchw(image, height, width, plane, output),
            OpenCvImage::Owned(image) => self.mat_into_nchw(image, height, width, plane, output),
        }
    }

    fn mat_into_nchw(
        &self,
        image: &impl ToInputArray,
        height: usize,
        width: usize,
        plane: usize,
        output: &mut [f32],
    ) -> RameResult<()> {
        let mut channels = Vector::<Mat>::new();
        let coefficients = NormalizationCoefficients::from(self.normalize);
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
                    coefficients.scale[channel],
                    coefficients.bias[channel],
                )
                .map_err(PreprocessError::from)?;
        }

        Ok(())
    }
}

fn metadata_tensor(data: Vec<f32>, len: usize) -> RameResult<Tensor> {
    Tensor::from_vec(data, &[len, 2]).map_err(|err| {
        PreprocessError::Backend {
            backend: "candle",
            message: err.to_string(),
        }
        .into()
    })
}

#[derive(Debug, Clone, Copy)]
struct NormalizationCoefficients {
    scale: [f64; 3],
    bias: [f64; 3],
}

impl From<Option<NormalizeImage>> for NormalizationCoefficients {
    fn from(normalize: Option<NormalizeImage>) -> Self {
        let Some(normalize) = normalize else {
            return Self {
                scale: [1.0, 1.0, 1.0],
                bias: [0.0, 0.0, 0.0],
            };
        };

        Self {
            scale: [
                (normalize.scale / normalize.std[0]) as f64,
                (normalize.scale / normalize.std[1]) as f64,
                (normalize.scale / normalize.std[2]) as f64,
            ],
            bias: [
                (-normalize.mean[0] / normalize.std[0]) as f64,
                (-normalize.mean[1] / normalize.std[1]) as f64,
                (-normalize.mean[2] / normalize.std[2]) as f64,
            ],
        }
    }
}

fn ensure_size(image: &OpenCvImage<'_>, expected: Size) -> RameResult<()> {
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
    use crate::preprocess::pipeline::PreprocessOp;
    use crate::tensor::Tensor;
    use ndarray::ArrayD;

    use crate::preprocess::vision::opencv::state::{OpenCvVisionBatch, OpenCvVisionData};
    use crate::preprocess::vision::{NormalizeImage, ToTensor};

    #[test]
    fn converts_images_to_nchw_tensor() {
        let images = [Image::from_rgb8(2, 1, vec![255, 0, 64, 32, 128, 255]).unwrap()];
        let image_views = images.iter().map(Image::as_view).collect::<Vec<_>>();
        let batch = OpenCvVisionBatch::new(&image_views).unwrap();
        let output = ToTensor::nchw()
            .forward(OpenCvVisionData::ImageBatch(batch))
            .unwrap();
        let OpenCvVisionData::TensorBatch(output) = output else {
            panic!("expected tensor batch");
        };

        assert_eq!(output.tensor.dims(), &[1, 3, 1, 2]);
        let tensor = tensor_to_array(output.tensor);
        assert_eq!(tensor[[0, 0, 0, 0]], 255.0);
        assert_eq!(tensor[[0, 1, 0, 0]], 0.0);
        assert_eq!(tensor[[0, 2, 0, 0]], 64.0);
        assert_eq!(tensor[[0, 0, 0, 1]], 32.0);
        assert_eq!(tensor[[0, 1, 0, 1]], 128.0);
        assert_eq!(tensor[[0, 2, 0, 1]], 255.0);
    }

    #[test]
    fn applies_normalization_while_converting_to_nchw_tensor() {
        let images = [
            Image::from_rgb8(2, 1, vec![255, 0, 64, 32, 128, 255]).unwrap(),
            Image::from_rgb8(2, 1, vec![0, 255, 128, 255, 64, 32]).unwrap(),
        ];

        let image_views = images.iter().map(Image::as_view).collect::<Vec<_>>();
        let normalize = NormalizeImage::new(0.5, [0.1, 0.2, 0.3], [1.0, 2.0, 4.0]);
        let batch = OpenCvVisionBatch::new(&image_views).unwrap();

        let output = ToTensor::nchw()
            .normalize(normalize)
            .forward(OpenCvVisionData::ImageBatch(batch))
            .unwrap();
        let OpenCvVisionData::TensorBatch(output) = output else {
            panic!("expected tensor batch");
        };

        let expected = ndarray::Array4::from_shape_vec(
            (2, 3, 1, 2),
            vec![
                255.0 * 0.5 - 0.1,
                32.0 * 0.5 - 0.1,
                (0.0 * 0.5 - 0.2) / 2.0,
                (128.0 * 0.5 - 0.2) / 2.0,
                (64.0 * 0.5 - 0.3) / 4.0,
                (255.0 * 0.5 - 0.3) / 4.0,
                0.0 * 0.5 - 0.1,
                255.0 * 0.5 - 0.1,
                (255.0 * 0.5 - 0.2) / 2.0,
                (64.0 * 0.5 - 0.2) / 2.0,
                (128.0 * 0.5 - 0.3) / 4.0,
                (32.0 * 0.5 - 0.3) / 4.0,
            ],
        )
        .unwrap();

        assert_eq!(output.tensor.dims(), expected.shape());
        assert_eq!(output.image_shapes.dims(), &[2, 2]);
        assert_eq!(output.scale_factors.dims(), &[2, 2]);

        let tensor = tensor_to_array(output.tensor);
        for (index, (left, right)) in tensor.iter().zip(expected.iter()).enumerate() {
            let diff = (left - right).abs();
            assert!(
                diff < 1e-5,
                "tensor mismatch at {index}: left={left}, right={right}, diff={diff}"
            );
        }
    }

    fn tensor_to_array(tensor: Tensor) -> ArrayD<f32> {
        tensor.to_array().unwrap()
    }
}
