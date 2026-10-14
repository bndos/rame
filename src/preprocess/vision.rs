pub use rame_core::preprocess::vision::{
    Interpolation, NormalizeImage, Resize, ResizeMode, TensorLayout, ToTensor, VisionBatchOutput,
};

#[cfg(feature = "opencv")]
pub use crate::backends::opencv::OpenCvVisionBackend;

#[cfg(feature = "opencv")]
pub type DefaultVisionBackend = OpenCvVisionBackend;

#[cfg(feature = "opencv")]
pub type VisionPipeline = crate::preprocess::pipeline::PreprocessPipeline<DefaultVisionBackend>;

#[cfg(feature = "opencv")]
pub type VisionPipelineBuilder =
    crate::preprocess::pipeline::PreprocessPipelineBuilder<DefaultVisionBackend>;

#[cfg(feature = "opencv")]
pub fn pipeline(config: crate::preprocess::PreprocessConfig) -> VisionPipelineBuilder {
    crate::preprocess::pipeline::PreprocessPipelineBuilder::new(DefaultVisionBackend::from(config))
}

#[cfg(all(test, feature = "opencv"))]
mod tests {
    use crate::image::Image;
    use crate::preprocess::vision::{Interpolation, NormalizeImage, Resize, ToTensor};

    #[test]
    fn default_pipeline_runs_vision_ops() {
        let image = Image::from_rgb8(1, 1, vec![255, 127, 0]).unwrap();
        let pipeline = super::pipeline(crate::preprocess::PreprocessConfig::default())
            .add_op(Resize::fixed_square(2, Interpolation::Cubic))
            .add_op(ToTensor::nchw().normalize(NormalizeImage::scale(NormalizeImage::INV_255)))
            .compile();

        let image_view = image.as_view();
        let output = pipeline.process_many(&[image_view]).unwrap();

        assert_eq!(output.tensor.dims(), &[1, 3, 2, 2]);
        assert_eq!(output.scale_factors.dims(), &[1, 2]);
        let scale_factors = crate::tensor::to_array::<f32>(&output.scale_factors).unwrap();
        assert_eq!(scale_factors[[0, 0]], 2.0);
        assert_eq!(scale_factors[[0, 1]], 2.0);
        let tensor = crate::tensor::to_array::<f32>(&output.tensor).unwrap();
        assert_eq!(tensor[[0, 0, 0, 0]], 1.0);
    }
}
