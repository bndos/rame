use ndarray::ArrayView2;

use crate::RameResult;
use crate::layout::{Geometry, LayoutRegion, LayoutResult, Rect};
use crate::models::pp_doclayout::plus::boxes::BatchedBoxes;
use crate::models::pp_doclayout::plus::labels::label_for_class_id;
use crate::runtime::{DecodeBatch, Decoder};

/// Decodes PP-DocLayout Plus detection outputs into layout regions.
///
/// The artifact config names the packed boxes output and the per-image
/// `boxes_num` output used to split batched detections before decoding. Box rows
/// follow PaddleOCR's postprocessed layout:
/// `[class_id, score, x_min, y_min, x_max, y_max]`.
/// Source: <https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/postprocess/picodet_postprocess.py#L251-L269>
#[derive(Debug, Clone)]
pub struct PpDocLayoutPlusDecoder {
    boxes_output_name: String,
    boxes_num_output_name: String,
}

impl PpDocLayoutPlusDecoder {
    pub fn new(
        boxes_output_name: impl Into<String>,
        boxes_num_output_name: impl Into<String>,
    ) -> Self {
        Self {
            boxes_output_name: boxes_output_name.into(),
            boxes_num_output_name: boxes_num_output_name.into(),
        }
    }
}

impl Decoder for PpDocLayoutPlusDecoder {
    type Output = LayoutResult;
    type Context = ();

    fn decode_batch(&self, batch: DecodeBatch<'_, Self::Context>) -> RameResult<Vec<Self::Output>> {
        if batch.len == 0 {
            return Ok(Vec::new());
        }

        let batched_boxes = BatchedBoxes::from_outputs(
            batch.outputs,
            &self.boxes_output_name,
            &self.boxes_num_output_name,
        )?;
        let mut results = Vec::with_capacity(batched_boxes.len());

        for index in 0..batched_boxes.len() {
            results.push(decode_boxes(batched_boxes.item(index)));
        }

        Ok(results)
    }
}

fn decode_boxes(boxes: ArrayView2<'_, f32>) -> LayoutResult {
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

    LayoutResult::new(regions)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use crate::layout::{Geometry, LayoutLabel, Rect};
    use crate::models::pp_doclayout::plus::decoder::PpDocLayoutPlusDecoder;
    use crate::runtime::{DecodeBatch, Decoder};
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
        let outputs = outputs_with_counts(boxes, vec![2]);

        let decoder = PpDocLayoutPlusDecoder::new("boxes", "boxes_num");
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
        let outputs = outputs_with_counts(boxes, vec![1]);

        let decoder = PpDocLayoutPlusDecoder::new("boxes", "boxes_num");
        let result = decoder.decode(&outputs, &()).unwrap();

        assert_eq!(result.regions[0].label, LayoutLabel::Text);
    }

    #[test]
    fn decodes_batched_pp_doclayout_plus_boxes() {
        let boxes = Array2::from_shape_vec(
            (4, 6),
            vec![
                0.0, 0.99, 1.0, 2.0, 3.0, 4.0, //
                8.0, 0.75, 5.0, 6.0, 7.0, 8.0, //
                2.0, 0.88, 9.0, 10.0, 11.0, 12.0, //
                1.0, 0.77, 13.0, 14.0, 15.0, 16.0,
            ],
        )
        .unwrap();
        let outputs = outputs_with_counts(boxes, vec![2, 2]);

        let decoder = PpDocLayoutPlusDecoder::new("boxes", "boxes_num");
        let results = decoder
            .decode_batch(DecodeBatch {
                len: 2,
                outputs: &outputs,
                contexts: &[(), ()],
            })
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].regions.len(), 2);
        assert_eq!(results[0].regions[0].label, LayoutLabel::Title);
        assert_eq!(results[0].regions[1].label, LayoutLabel::Table);
        assert_eq!(results[1].regions.len(), 2);
        assert_eq!(results[1].regions[0].label, LayoutLabel::Text);
        assert_eq!(results[1].regions[1].label, LayoutLabel::Image);
    }

    #[test]
    fn rejects_batched_boxes_when_counts_do_not_match_rows() {
        let boxes = Array2::from_shape_vec(
            (3, 6),
            vec![
                0.0, 0.99, 1.0, 2.0, 3.0, 4.0, //
                8.0, 0.75, 5.0, 6.0, 7.0, 8.0, //
                2.0, 0.88, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let outputs = outputs_with_counts(boxes, vec![2, 2]);

        let decoder = PpDocLayoutPlusDecoder::new("boxes", "boxes_num");
        let err = decoder
            .decode_batch(DecodeBatch {
                len: 2,
                outputs: &outputs,
                contexts: &[(), ()],
            })
            .unwrap_err();

        assert!(err.to_string().contains("4 rows from `boxes_num`"));
    }

    #[test]
    fn rejects_non_f32_boxes() {
        let boxes = Array2::from_shape_vec((1, 6), vec![0_i64, 1, 2, 3, 4, 5]).unwrap();
        let mut outputs = TensorMap::new();
        outputs.insert("boxes".to_string(), TensorValue::I64(boxes.into_dyn()));
        outputs.insert(
            "boxes_num".to_string(),
            TensorValue::I32(Array1::from_vec(vec![1]).into_dyn()),
        );

        let decoder = PpDocLayoutPlusDecoder::new("boxes", "boxes_num");
        let err = decoder.decode(&outputs, &()).unwrap_err();

        assert!(err.to_string().contains("expected f32"));
    }

    #[test]
    fn rejects_invalid_boxes_shape() {
        let boxes = Array2::from_shape_vec((1, 5), vec![2.0, 0.99, 1.0, 2.0, 3.0]).unwrap();
        let outputs = outputs_with_counts(boxes, vec![1]);

        let decoder = PpDocLayoutPlusDecoder::new("boxes", "boxes_num");
        let err = decoder.decode(&outputs, &()).unwrap_err();

        assert!(err.to_string().contains("expected [N, 6]"));
    }

    #[test]
    fn requires_configured_boxes_output_name() {
        let boxes = Array2::from_shape_vec((1, 6), vec![2.0, 0.99, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut outputs = TensorMap::new();
        outputs.insert("other".to_string(), TensorValue::F32(boxes.into_dyn()));
        outputs.insert(
            "boxes_num".to_string(),
            TensorValue::I32(Array1::from_vec(vec![1]).into_dyn()),
        );

        let decoder = PpDocLayoutPlusDecoder::new("boxes", "boxes_num");
        let err = decoder.decode(&outputs, &()).unwrap_err();

        assert!(err.to_string().contains("missing tensor `boxes`"));
    }

    fn outputs_with_counts(boxes: Array2<f32>, counts: Vec<i32>) -> TensorMap {
        let mut outputs = TensorMap::new();
        outputs.insert("boxes".to_string(), TensorValue::F32(boxes.into_dyn()));
        outputs.insert(
            "boxes_num".to_string(),
            TensorValue::I32(Array1::from_vec(counts).into_dyn()),
        );
        outputs
    }
}
