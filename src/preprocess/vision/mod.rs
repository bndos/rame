#[cfg(feature = "opencv")]
mod opencv;
mod ops;
mod output;

#[cfg(feature = "opencv")]
pub use opencv::OpenCvVisionBackend;
pub use ops::{Interpolation, NormalizeImage, Resize, ResizeMode, TensorLayout, ToTensor};
pub use output::VisionBatchOutput;

#[cfg(feature = "opencv")]
pub type DefaultVisionBackend = OpenCvVisionBackend;

#[cfg(feature = "opencv")]
pub type VisionPipeline = crate::preprocess::pipeline::PreprocessPipeline<DefaultVisionBackend>;

#[cfg(feature = "opencv")]
pub fn pipeline() -> VisionPipeline {
    crate::preprocess::pipeline::PreprocessPipeline::new(DefaultVisionBackend::default())
}

#[cfg(all(test, feature = "opencv"))]
mod tests {
    use crate::image::Image;
    use crate::preprocess::vision::{Interpolation, NormalizeImage, Resize, ToTensor};

    #[test]
    fn default_pipeline_runs_vision_ops() {
        let image = Image::from_rgb8(1, 1, vec![255, 127, 0]).unwrap();
        let pipeline = super::pipeline()
            .add_op(Resize::fixed_square(2, Interpolation::Cubic))
            .add_op(ToTensor::nchw().normalize(NormalizeImage::scale(NormalizeImage::INV_255)))
            .compile();

        let image_view = image.as_view();
        let output = pipeline.process_many(&[image_view]).unwrap();

        assert_eq!(output.tensor.shape(), &[1, 3, 2, 2]);
        assert_eq!(output.scale_factors.shape(), &[1, 2]);
        assert_eq!(output.scale_factors[[0, 0]], 2.0);
        assert_eq!(output.scale_factors[[0, 1]], 2.0);
        assert_eq!(output.tensor[[0, 0, 0, 0]], 1.0);
    }
}
