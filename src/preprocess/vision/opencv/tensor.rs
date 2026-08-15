use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp};
use crate::preprocess::vision::ToTensor;
use crate::preprocess::vision::opencv::OpenCvVisionBackend;
use crate::preprocess::vision::opencv::cpu;
use crate::preprocess::vision::opencv::state::OpenCvVisionData;

impl PreprocessOp<OpenCvVisionBackend> for ToTensor {
    fn forward<'a>(
        &self,
        data: <OpenCvVisionBackend as PreprocessBackend>::Data<'a>,
    ) -> RameResult<<OpenCvVisionBackend as PreprocessBackend>::Data<'a>> {
        let batch = data.into_image_batch()?;
        let output = match &*batch.device {
            candle_core::Device::Cpu => cpu::to_tensor(self, batch)?,
            candle_core::Device::Cuda(_) => {
                return Err(PreprocessError::UnsupportedBackendOp {
                    backend: "OpenCV CUDA",
                    op: "ToTensor",
                }
                .into());
            }
            candle_core::Device::Metal(_) => {
                return Err(PreprocessError::UnsupportedBackendOp {
                    backend: "OpenCV Metal",
                    op: "ToTensor",
                }
                .into());
            }
        };

        Ok(OpenCvVisionData::TensorBatch(output))
    }
}

#[cfg(test)]
mod tests {
    use crate::image::Image;
    use crate::preprocess::pipeline::PreprocessOp;
    use crate::tensor::{Device, Tensor};
    use ndarray::ArrayD;

    use crate::preprocess::vision::opencv::state::{OpenCvVisionBatch, OpenCvVisionData};
    use crate::preprocess::vision::{NormalizeImage, ToTensor};

    #[test]
    fn converts_images_to_nchw_tensor() {
        let images = [Image::from_rgb8(2, 1, vec![255, 0, 64, 32, 128, 255]).unwrap()];
        let image_views = images.iter().map(Image::as_view).collect::<Vec<_>>();
        let batch = OpenCvVisionBatch::new(&image_views, Device::cpu()).unwrap();
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
        let batch = OpenCvVisionBatch::new(&image_views, Device::cpu()).unwrap();

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
