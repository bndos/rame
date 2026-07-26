use ndarray::{Array2, Array4, Axis, s};

use crate::RameResult;
use crate::image::Image;
use crate::models::ModelError;
use crate::models::pp_doclayout::plus::onnx::{Inputs, Preprocess};
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

    fn process(&self, image: &Image) -> RameResult<Processed<Self::Context>> {
        let output = crate::preprocess::vision::pipeline()
            .add_op(self.preprocess.resize)
            .add_op(self.preprocess.normalize)
            .add_op(self.preprocess.permute)
            .process(image)?;

        let tensor_shape = output.tensor.shape();
        let image_height = tensor_shape[2] as f32;
        let image_width = tensor_shape[3] as f32;

        let mut inputs = TensorMap::new();
        inputs.insert(
            self.inputs.image.clone(),
            TensorValue::F32(output.tensor.into_dyn()),
        );
        inputs.insert(
            self.inputs.im_shape.clone(),
            TensorValue::F32(row_tensor([image_height, image_width])),
        );
        inputs.insert(
            self.inputs.scale_factor.clone(),
            TensorValue::F32(row_tensor(output.scale_factor)),
        );

        Ok(Processed {
            inputs,
            context: (),
        })
    }
}

fn row_tensor(values: [f32; 2]) -> ndarray::ArrayD<f32> {
    Array2::from_shape_vec((1, 2), values.to_vec())
        .expect("two values always fit a [1, 2] tensor")
        .into_dyn()
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

        let processed = processor.process(&image).unwrap();

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
}
