#[cfg(feature = "opencv")]
mod opencv;
mod ops;
mod output;

#[cfg(feature = "opencv")]
pub use opencv::OpenCvVisionBackend;
pub use ops::{Interpolation, NormalizeAndPermute, PixelScale, Resize, ResizeMode, TensorLayout};
pub use output::VisionTensorOutput;

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
    use crate::preprocess::vision::{
        Interpolation, NormalizeAndPermute, PixelScale, Resize, TensorLayout,
    };

    #[test]
    fn default_pipeline_runs_vision_ops() {
        let image = Image::from_rgb8(1, 1, vec![255, 127, 0]).unwrap();
        let pipeline = super::pipeline()
            .add_op(Resize::fixed_square(2, Interpolation::Cubic))
            .add_op(NormalizeAndPermute::new(
                PixelScale::OneOver255,
                TensorLayout::Nchw,
            ));

        let output = pipeline.process(&image).unwrap();

        assert_eq!(output.tensor.shape(), &[1, 3, 2, 2]);
        assert_eq!(output.scale_factor, [2.0, 2.0]);
        assert_eq!(output.tensor[[0, 0, 0, 0]], 1.0);
    }
}
