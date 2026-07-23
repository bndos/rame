use ndarray::{ArrayView2, Ix2};

use crate::RameResult;
use crate::layout::{Geometry, LayoutRegion, LayoutResult, Rect};
use crate::models::ModelError;
use crate::models::pp_doclayout::plus::labels::label_for_class_id;
use crate::runtime::Decoder;
use crate::tensor::{TensorMap, TensorValue};

/// Decodes PP-DocLayout Plus boxes shaped `[N, 6]`.
///
/// Rows follow PaddleOCR's postprocessed layout:
/// `[class_id, score, x_min, y_min, x_max, y_max]`.
/// Source: <https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/postprocess/picodet_postprocess.py#L251-L269>
#[derive(Debug, Clone)]
pub struct PpDocLayoutPlusDecoder {
    boxes_output_name: String,
}

impl PpDocLayoutPlusDecoder {
    pub fn new(boxes_output_name: impl Into<String>) -> Self {
        Self {
            boxes_output_name: boxes_output_name.into(),
        }
    }
}

impl Decoder for PpDocLayoutPlusDecoder {
    type Output = LayoutResult;
    type Context = ();

    fn decode(&self, outputs: &TensorMap, _context: &Self::Context) -> RameResult<Self::Output> {
        let boxes = require_boxes_tensor(outputs, &self.boxes_output_name)?;
        let mut regions = Vec::new();

        for row in boxes.outer_iter() {
            let class_id = row[0] as i64;
            let label = label_for_class_id(class_id);

            regions.push(LayoutRegion {
                label,
                score: row[1],
                geometry: Geometry::Rect(Rect::new(row[2], row[3], row[4], row[5])),
                reading_order: None,
            });
        }

        Ok(LayoutResult::new(regions))
    }
}

fn require_boxes_tensor<'a>(outputs: &'a TensorMap, name: &str) -> RameResult<ArrayView2<'a, f32>> {
    let tensor = outputs
        .get(name)
        .ok_or_else(|| ModelError::MissingTensor(name.to_string()))?;

    let TensorValue::F32(tensor) = tensor else {
        return Err(ModelError::InvalidTensorType {
            name: name.to_string(),
            expected: "f32".to_string(),
            actual: tensor.kind().to_string(),
        }
        .into());
    };

    if tensor.shape().len() != 2 || tensor.shape()[1] != 6 {
        return Err(ModelError::InvalidTensorShape {
            name: name.to_string(),
            expected: "[N, 6]".to_string(),
            actual: tensor.shape().to_vec(),
        }
        .into());
    }

    tensor.view().into_dimensionality::<Ix2>().map_err(|_| {
        ModelError::InvalidTensorShape {
            name: name.to_string(),
            expected: "[N, 6]".to_string(),
            actual: tensor.shape().to_vec(),
        }
        .into()
    })
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use crate::layout::{Geometry, LayoutLabel, Rect};
    use crate::models::pp_doclayout::plus::decoder::PpDocLayoutPlusDecoder;
    use crate::runtime::Decoder;
    use crate::tensor::{TensorMap, TensorValue};

    #[test]
    fn decodes_pp_doclayout_plus_boxes() {
        let boxes = Array2::from_shape_vec(
            (2, 6),
            vec![
                0.0, 0.99, 1.0, 2.0, 3.0, 4.0, //
                8.0, 0.75, 5.0, 6.0, 7.0, 8.0,
            ],
        )
        .unwrap();
        let mut outputs = TensorMap::new();
        outputs.insert("boxes".to_string(), TensorValue::F32(boxes.into_dyn()));

        let decoder = PpDocLayoutPlusDecoder::new("boxes");
        let result = decoder.decode(&outputs, &()).unwrap();

        assert_eq!(result.regions.len(), 2);
        assert_eq!(result.regions[0].label, LayoutLabel::Title);
        assert_eq!(result.regions[0].score, 0.99);
        assert_eq!(
            result.regions[0].geometry,
            Geometry::Rect(Rect::new(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(result.regions[1].label, LayoutLabel::Table);
    }

    #[test]
    fn decodes_pp_doclayout_plus_text_class_id() {
        let boxes = Array2::from_shape_vec((1, 6), vec![2.0, 0.99, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut outputs = TensorMap::new();
        outputs.insert("boxes".to_string(), TensorValue::F32(boxes.into_dyn()));

        let decoder = PpDocLayoutPlusDecoder::new("boxes");
        let result = decoder.decode(&outputs, &()).unwrap();

        assert_eq!(result.regions[0].label, LayoutLabel::Text);
    }

    #[test]
    fn rejects_non_f32_boxes() {
        let boxes = Array2::from_shape_vec((1, 6), vec![0_i64, 1, 2, 3, 4, 5]).unwrap();
        let mut outputs = TensorMap::new();
        outputs.insert("boxes".to_string(), TensorValue::I64(boxes.into_dyn()));

        let decoder = PpDocLayoutPlusDecoder::new("boxes");
        let err = decoder.decode(&outputs, &()).unwrap_err();

        assert!(err.to_string().contains("expected f32"));
    }

    #[test]
    fn rejects_invalid_boxes_shape() {
        let boxes = Array2::from_shape_vec((1, 5), vec![2.0, 0.99, 1.0, 2.0, 3.0]).unwrap();
        let mut outputs = TensorMap::new();
        outputs.insert("boxes".to_string(), TensorValue::F32(boxes.into_dyn()));

        let decoder = PpDocLayoutPlusDecoder::new("boxes");
        let err = decoder.decode(&outputs, &()).unwrap_err();

        assert!(err.to_string().contains("expected [N, 6]"));
    }

    #[test]
    fn requires_configured_boxes_output_name() {
        let boxes = Array2::from_shape_vec((1, 6), vec![2.0, 0.99, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut outputs = TensorMap::new();
        outputs.insert("other".to_string(), TensorValue::F32(boxes.into_dyn()));

        let decoder = PpDocLayoutPlusDecoder::new("boxes");
        let err = decoder.decode(&outputs, &()).unwrap_err();

        assert!(err.to_string().contains("missing tensor `boxes`"));
    }
}
