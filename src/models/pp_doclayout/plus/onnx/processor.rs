use crate::RameResult;
use crate::image::ImageView;
use crate::models::pp_doclayout::plus::onnx::{Inputs, Preprocess};
use crate::preprocess::vision::VisionPipeline;
use crate::runtime::{ProcessedBatch, Processor};
use crate::tensor::TensorMap;

#[derive(Debug)]
#[doc(hidden)]
pub struct PpDocLayoutPlusOnnxProcessor {
    inputs: Inputs,
    pipeline: VisionPipeline,
}

impl PpDocLayoutPlusOnnxProcessor {
    pub fn new(inputs: Inputs, preprocess: Preprocess) -> Self {
        let pipeline = crate::preprocess::vision::pipeline()
            .add_op(preprocess.resize)
            .add_op(preprocess.tensor)
            .compile();

        Self { inputs, pipeline }
    }
}

impl Processor for PpDocLayoutPlusOnnxProcessor {
    type Source<'a> = ImageView<'a>;
    type Context = ();

    fn process_many<'a>(
        &self,
        images: &'a [Self::Source<'a>],
    ) -> RameResult<ProcessedBatch<Self::Context>> {
        if images.is_empty() {
            return Ok(ProcessedBatch {
                len: 0,
                inputs: TensorMap::new(),
                contexts: Vec::new(),
            });
        }

        let output = self.pipeline.process_many(images)?;

        Ok(ProcessedBatch {
            len: output.len(),
            contexts: vec![(); output.len()],
            inputs: bind_inputs(&self.inputs, output),
        })
    }
}

fn bind_inputs(inputs: &Inputs, output: crate::preprocess::vision::VisionBatchOutput) -> TensorMap {
    let mut tensors = TensorMap::new();
    tensors.insert(inputs.image.clone(), output.tensor);
    tensors.insert(inputs.im_shape.clone(), output.image_shapes);
    tensors.insert(inputs.scale_factor.clone(), output.scale_factors);
    tensors
}

#[cfg(test)]
mod tests {
    use crate::image::Image;
    use crate::models::pp_doclayout::plus::onnx::{Inputs, Preprocess};
    use crate::preprocess::vision::{Interpolation, Resize};
    use crate::runtime::Processor;
    use crate::tensor::Tensor;
    use candle_core::DType;
    use ndarray::ArrayD;

    use super::PpDocLayoutPlusOnnxProcessor;

    #[test]
    fn creates_paddle_onnx_inputs() {
        let image = Image::from_rgb8(1, 1, vec![255, 127, 0]).unwrap();
        let processor = PpDocLayoutPlusOnnxProcessor::new(
            Inputs::default(),
            Preprocess {
                resize: Resize::fixed_square(2, Interpolation::Cubic),
                ..Preprocess::default()
            },
        );

        let image_view = image.as_view();
        let processed = processor.process_many(&[image_view]).unwrap();

        assert_eq!(processed.len, 1);
        assert_eq!(processed.contexts.len(), 1);
        assert_eq!(processed.inputs.len(), 3);
        let image = f32_array(&processed.inputs["image"]);
        let im_shape = f32_array(&processed.inputs["im_shape"]);
        let scale_factor = f32_array(&processed.inputs["scale_factor"]);

        assert_eq!(processed.inputs["image"].dtype(), DType::F32);
        assert_eq!(image.shape(), &[1, 3, 2, 2]);
        assert_eq!(image[[0, 0, 0, 0]], 1.0);
        assert_eq!(im_shape.shape(), &[1, 2]);
        assert_eq!(im_shape[[0, 0]], 2.0);
        assert_eq!(im_shape[[0, 1]], 2.0);
        assert_eq!(scale_factor.shape(), &[1, 2]);
        assert_eq!(scale_factor[[0, 0]], 2.0);
        assert_eq!(scale_factor[[0, 1]], 2.0);
    }

    #[test]
    fn creates_batched_paddle_onnx_inputs() {
        let images = [
            Image::from_rgb8(1, 1, vec![255, 127, 0]).unwrap(),
            Image::from_rgb8(1, 1, vec![0, 127, 255]).unwrap(),
        ];
        let processor = PpDocLayoutPlusOnnxProcessor::new(
            Inputs::default(),
            Preprocess {
                resize: Resize::fixed_square(2, Interpolation::Cubic),
                ..Preprocess::default()
            },
        );

        let image_views = images.iter().map(Image::as_view).collect::<Vec<_>>();
        let processed = processor.process_many(&image_views).unwrap();

        assert_eq!(processed.len, 2);
        assert_eq!(processed.contexts.len(), 2);
        let image = f32_array(&processed.inputs["image"]);
        let im_shape = f32_array(&processed.inputs["im_shape"]);
        let scale_factor = f32_array(&processed.inputs["scale_factor"]);

        assert_eq!(processed.inputs["image"].dtype(), DType::F32);
        assert_eq!(image.shape(), &[2, 3, 2, 2]);
        assert_eq!(image[[0, 0, 0, 0]], 1.0);
        assert_eq!(image[[1, 2, 0, 0]], 1.0);
        assert_eq!(im_shape.shape(), &[2, 2]);
        assert_eq!(im_shape[[0, 0]], 2.0);
        assert_eq!(im_shape[[1, 1]], 2.0);
        assert_eq!(scale_factor.shape(), &[2, 2]);
        assert_eq!(scale_factor[[0, 0]], 2.0);
        assert_eq!(scale_factor[[1, 1]], 2.0);
    }

    fn f32_array(tensor: &Tensor) -> ArrayD<f32> {
        tensor.to_array().unwrap_or_else(|err| {
            panic!("expected f32 tensor: {err}");
        })
    }
}
