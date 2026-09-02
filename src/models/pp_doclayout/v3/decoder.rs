use ndarray::ArrayView2;

use crate::RameResult;
use crate::geometry::Rect;
use crate::layout::{Geometry, LayoutRegion, LayoutResult};
use crate::models::pp_doclayout::boxes::BatchedBoxes;
use crate::models::pp_doclayout::v3::labels::label_for_class_id;
use crate::runtime::{DecodeBatch, Decoder};

/// Decodes PP-DocLayoutV3 detection outputs into layout regions.
///
/// The official ONNX graph packs detections into `fetch_name_0` with seven
/// columns and emits per-image counts in `fetch_name_1`. Box rows are interpreted
/// as `[class_id, score, x_min, y_min, x_max, y_max, reading_order]`.
/// Source: <https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/blob/main/inference.yml>
#[derive(Debug, Clone)]
pub struct PpDocLayoutV3Decoder {
    boxes_output_name: String,
    boxes_num_output_name: String,
}

impl PpDocLayoutV3Decoder {
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

impl Decoder for PpDocLayoutV3Decoder {
    type Output = LayoutResult;
    type Context = ();

    fn decode_batch(&self, batch: DecodeBatch<'_, Self::Context>) -> RameResult<Vec<Self::Output>> {
        let batched_boxes = BatchedBoxes::from_outputs(
            batch.outputs,
            &self.boxes_output_name,
            &self.boxes_num_output_name,
            7,
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
    let mut indices = (0..boxes.nrows()).collect::<Vec<_>>();
    indices.sort_by(|&left, &right| boxes[[left, 6]].total_cmp(&boxes[[right, 6]]));

    for index in indices {
        let row = boxes.row(index);
        let class_id = row[0] as i64;
        let label = label_for_class_id(class_id);

        regions.push(LayoutRegion {
            label,
            score: row[1],
            geometry: Geometry::Rect(Rect::new(row[2], row[3], row[4], row[5])),
            reading_order: Some(row[6] as usize),
        });
    }

    LayoutResult::new(regions)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use crate::geometry::Rect;
    use crate::layout::{Geometry, LayoutLabel};
    use crate::models::pp_doclayout::v3::decoder::PpDocLayoutV3Decoder;
    use crate::runtime::{DecodeBatch, Decoder};
    use crate::tensor::{Tensor, TensorMap};

    #[test]
    fn decodes_pp_doclayout_v3_boxes() {
        let boxes = Array2::from_shape_vec(
            (2, 7),
            vec![
                21.0, 0.75, 5.0, 6.0, 7.0, 8.0, 1.0, //
                6.0, 0.99, 1.0, 2.0, 3.0, 4.0, 0.0,
            ],
        )
        .unwrap();
        let outputs = outputs_with_counts(boxes, vec![2]);

        let decoder = PpDocLayoutV3Decoder::new("boxes", "boxes_num");
        let result = decoder.decode(&outputs, &()).unwrap();

        assert_eq!(result.regions.len(), 2);
        assert_eq!(result.regions[0].label, LayoutLabel::Title);
        assert_eq!(result.regions[0].score, 0.99);
        assert_eq!(result.regions[0].reading_order, Some(0));
        assert_eq!(
            result.regions[0].geometry,
            Geometry::Rect(Rect::new(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(result.regions[1].label, LayoutLabel::Table);
        assert_eq!(result.regions[1].reading_order, Some(1));
    }

    #[test]
    fn decodes_pp_doclayout_v3_text_class_id() {
        let boxes =
            Array2::from_shape_vec((1, 7), vec![22.0, 0.99, 1.0, 2.0, 3.0, 4.0, 0.0]).unwrap();
        let outputs = outputs_with_counts(boxes, vec![1]);

        let decoder = PpDocLayoutV3Decoder::new("boxes", "boxes_num");
        let result = decoder.decode(&outputs, &()).unwrap();

        assert_eq!(result.regions[0].label, LayoutLabel::Text);
        assert_eq!(result.regions[0].reading_order, Some(0));
    }

    #[test]
    fn decodes_batched_pp_doclayout_v3_boxes() {
        let boxes = Array2::from_shape_vec(
            (4, 7),
            vec![
                21.0, 0.75, 5.0, 6.0, 7.0, 8.0, 1.0, //
                6.0, 0.99, 1.0, 2.0, 3.0, 4.0, 0.0, //
                14.0, 0.77, 13.0, 14.0, 15.0, 16.0, 3.0, //
                22.0, 0.88, 9.0, 10.0, 11.0, 12.0, 2.0,
            ],
        )
        .unwrap();
        let outputs = outputs_with_counts(boxes, vec![2, 2]);

        let decoder = PpDocLayoutV3Decoder::new("boxes", "boxes_num");
        let results = decoder
            .decode_batch(DecodeBatch {
                outputs: &outputs,
                contexts: &[(), ()],
            })
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].regions.len(), 2);
        assert_eq!(results[0].regions[0].label, LayoutLabel::Title);
        assert_eq!(results[0].regions[1].label, LayoutLabel::Table);
        assert_eq!(results[0].regions[0].reading_order, Some(0));
        assert_eq!(results[0].regions[1].reading_order, Some(1));
        assert_eq!(results[1].regions.len(), 2);
        assert_eq!(results[1].regions[0].label, LayoutLabel::Text);
        assert_eq!(results[1].regions[1].label, LayoutLabel::Image);
        assert_eq!(results[1].regions[0].reading_order, Some(2));
        assert_eq!(results[1].regions[1].reading_order, Some(3));
    }

    #[test]
    fn rejects_batched_boxes_when_counts_do_not_match_rows() {
        let boxes = Array2::from_shape_vec(
            (3, 7),
            vec![
                6.0, 0.99, 1.0, 2.0, 3.0, 4.0, 0.0, //
                21.0, 0.75, 5.0, 6.0, 7.0, 8.0, 1.0, //
                22.0, 0.88, 9.0, 10.0, 11.0, 12.0, 2.0,
            ],
        )
        .unwrap();
        let outputs = outputs_with_counts(boxes, vec![2, 2]);

        let decoder = PpDocLayoutV3Decoder::new("boxes", "boxes_num");
        let err = decoder
            .decode_batch(DecodeBatch {
                outputs: &outputs,
                contexts: &[(), ()],
            })
            .unwrap_err();

        assert!(err.to_string().contains("4 rows from `boxes_num`"));
    }

    #[test]
    fn rejects_non_f32_boxes() {
        let boxes = Array2::from_shape_vec((1, 7), vec![0_i64, 1, 2, 3, 4, 5, 6]).unwrap();
        let mut outputs = TensorMap::new();
        outputs.insert("boxes".to_string(), tensor(boxes.into_dyn()));
        outputs.insert(
            "boxes_num".to_string(),
            tensor(Array1::from_vec(vec![1]).into_dyn()),
        );

        let decoder = PpDocLayoutV3Decoder::new("boxes", "boxes_num");
        let err = decoder.decode(&outputs, &()).unwrap_err();

        assert!(err.to_string().contains("expected f32"));
    }

    #[test]
    fn rejects_invalid_boxes_shape() {
        let boxes = Array2::from_shape_vec((1, 5), vec![2.0, 0.99, 1.0, 2.0, 3.0]).unwrap();
        let outputs = outputs_with_counts(boxes, vec![1]);

        let decoder = PpDocLayoutV3Decoder::new("boxes", "boxes_num");
        let err = decoder.decode(&outputs, &()).unwrap_err();

        assert!(err.to_string().contains("expected [N, 7]"));
    }

    #[test]
    fn requires_configured_boxes_output_name() {
        let boxes =
            Array2::from_shape_vec((1, 7), vec![22.0, 0.99, 1.0, 2.0, 3.0, 4.0, 0.0]).unwrap();
        let mut outputs = TensorMap::new();
        outputs.insert("other".to_string(), tensor(boxes.into_dyn()));
        outputs.insert(
            "boxes_num".to_string(),
            tensor(Array1::from_vec(vec![1]).into_dyn()),
        );

        let decoder = PpDocLayoutV3Decoder::new("boxes", "boxes_num");
        let err = decoder.decode(&outputs, &()).unwrap_err();

        assert!(err.to_string().contains("missing tensor `boxes`"));
    }

    fn outputs_with_counts(boxes: Array2<f32>, counts: Vec<i32>) -> TensorMap {
        let mut outputs = TensorMap::new();
        outputs.insert("boxes".to_string(), tensor(boxes.into_dyn()));
        outputs.insert(
            "boxes_num".to_string(),
            tensor(Array1::from_vec(counts).into_dyn()),
        );
        outputs
    }

    fn tensor<T>(array: ndarray::ArrayD<T>) -> Tensor
    where
        T: candle_core::WithDType + Clone,
    {
        Tensor::from_array(array).unwrap()
    }
}
