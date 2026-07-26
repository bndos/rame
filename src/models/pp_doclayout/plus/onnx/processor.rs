use ndarray::Array2;

use crate::RameResult;
use crate::image::Image;
use crate::models::pp_doclayout::plus::onnx::{Inputs, Preprocess};
use crate::preprocess::vision::{NchwBatchBuilder, VisionTensorOutput};
use crate::runtime::{ProcessedBatch, Processor};
use crate::tensor::{TensorMap, TensorValue};

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct PpDocLayoutPlusOnnxProcessor {
    inputs: Inputs,
    preprocess: Preprocess,
}

impl PpDocLayoutPlusOnnxProcessor {
    pub fn new(inputs: Inputs, preprocess: Preprocess) -> Self {
        Self { inputs, preprocess }
    }
}

impl Processor for PpDocLayoutPlusOnnxProcessor {
    type Source = Image;
    type Context = ();

    fn process_many(&self, images: &[Image]) -> RameResult<ProcessedBatch<Self::Context>> {
        if images.is_empty() {
            return Ok(ProcessedBatch {
                len: 0,
                inputs: TensorMap::new(),
                contexts: Vec::new(),
            });
        }

        let pipeline = crate::preprocess::vision::pipeline()
            .add_op(self.preprocess.resize)
            .add_op(self.preprocess.normalize)
            .add_op(self.preprocess.permute);
        let mut batch = PpDocLayoutInputBatch::new(images.len());
        for image in images {
            batch.push(pipeline.process(image)?)?;
        }

        Ok(ProcessedBatch {
            len: images.len(),
            inputs: batch.finish(&self.inputs)?,
            contexts: vec![(); images.len()],
        })
    }
}

struct PpDocLayoutInputBatch {
    image: NchwBatchBuilder,
    im_shape: Array2<f32>,
    scale_factor: Array2<f32>,
}

impl PpDocLayoutInputBatch {
    fn new(len: usize) -> Self {
        Self {
            image: NchwBatchBuilder::new(len),
            im_shape: Array2::zeros((len, 2)),
            scale_factor: Array2::zeros((len, 2)),
        }
    }

    fn push(&mut self, output: VisionTensorOutput) -> RameResult<()> {
        let tensor_shape = output.tensor.shape();
        let image_height = tensor_shape[2] as f32;
        let image_width = tensor_shape[3] as f32;
        let scale_factor = output.scale_factor;
        let index = self.image.push(output.tensor)?;

        self.im_shape[[index, 0]] = image_height;
        self.im_shape[[index, 1]] = image_width;
        self.scale_factor[[index, 0]] = scale_factor[0];
        self.scale_factor[[index, 1]] = scale_factor[1];
        Ok(())
    }

    fn finish(self, inputs: &Inputs) -> RameResult<TensorMap> {
        let mut tensors = TensorMap::new();
        tensors.insert(
            inputs.image.clone(),
            TensorValue::F32(self.image.finish()?.into_dyn()),
        );
        tensors.insert(
            inputs.im_shape.clone(),
            TensorValue::F32(self.im_shape.into_dyn()),
        );
        tensors.insert(
            inputs.scale_factor.clone(),
            TensorValue::F32(self.scale_factor.into_dyn()),
        );
        Ok(tensors)
    }
}

#[cfg(test)]
mod tests {
    use crate::image::Image;
    use crate::models::pp_doclayout::plus::onnx::{Inputs, Preprocess};
    use crate::preprocess::vision::{Interpolation, Resize};
    use crate::runtime::Processor;
    use crate::tensor::TensorValue;

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

        let processed = processor
            .process_many(std::slice::from_ref(&image))
            .unwrap();

        assert_eq!(processed.len, 1);
        assert_eq!(processed.contexts.len(), 1);
        assert_eq!(processed.inputs.len(), 3);
        let TensorValue::F32(image) = &processed.inputs["image"] else {
            panic!("expected f32 image tensor");
        };
        let TensorValue::F32(im_shape) = &processed.inputs["im_shape"] else {
            panic!("expected f32 im_shape tensor");
        };
        let TensorValue::F32(scale_factor) = &processed.inputs["scale_factor"] else {
            panic!("expected f32 scale_factor tensor");
        };

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
        let images = vec![
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

        let processed = processor.process_many(&images).unwrap();

        assert_eq!(processed.len, 2);
        assert_eq!(processed.contexts.len(), 2);
        let TensorValue::F32(image) = &processed.inputs["image"] else {
            panic!("expected f32 image tensor");
        };
        let TensorValue::F32(im_shape) = &processed.inputs["im_shape"] else {
            panic!("expected f32 im_shape tensor");
        };
        let TensorValue::F32(scale_factor) = &processed.inputs["scale_factor"] else {
            panic!("expected f32 scale_factor tensor");
        };

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
}
