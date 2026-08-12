use ndarray::{ArrayView2, Ix2, s};

use crate::RameResult;
use crate::models::ModelError;
use crate::tensor::{TensorMap, TensorValue};

/// Per-image view of packed Paddle layout detection boxes.
///
/// Paddle layout models can pack batched detections into one output tensor and
/// report how many rows belong to each image separately. This helper validates
/// and splits that representation.
#[derive(Debug)]
pub(super) struct BatchedBoxes<'a> {
    boxes: ArrayView2<'a, f32>,
    offsets: Vec<usize>,
}

impl<'a> BatchedBoxes<'a> {
    /// Reads a packed boxes output and its per-image row counts.
    pub(super) fn from_outputs(
        outputs: &'a TensorMap,
        boxes_name: &str,
        boxes_num_name: &str,
        columns: usize,
    ) -> RameResult<Self> {
        let boxes = require_boxes_tensor(outputs, boxes_name, columns)?;
        let counts = require_boxes_num_tensor(outputs, boxes_num_name)?;
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0);

        let mut total_boxes = 0;
        for count in counts {
            total_boxes += count;
            offsets.push(total_boxes);
        }

        if boxes.shape()[0] != total_boxes {
            return Err(ModelError::InvalidTensorShape {
                name: boxes_name.to_string(),
                expected: format!("{} rows from `{boxes_num_name}`", total_boxes),
                actual: boxes.shape().to_vec(),
            }
            .into());
        }

        Ok(Self { boxes, offsets })
    }

    pub(super) fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub(super) fn item(&self, index: usize) -> ArrayView2<'_, f32> {
        let start = self.offsets[index];
        let end = self.offsets[index + 1];
        self.boxes.slice(s![start..end, ..])
    }
}

fn require_boxes_tensor<'a>(
    outputs: &'a TensorMap,
    name: &str,
    columns: usize,
) -> RameResult<ArrayView2<'a, f32>> {
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

    let expected = format!("[N, {columns}]");
    if tensor.shape().len() != 2 || tensor.shape()[1] != columns {
        return Err(ModelError::InvalidTensorShape {
            name: name.to_string(),
            expected,
            actual: tensor.shape().to_vec(),
        }
        .into());
    }

    tensor.view().into_dimensionality::<Ix2>().map_err(|_| {
        ModelError::InvalidTensorShape {
            name: name.to_string(),
            expected: format!("[N, {columns}]"),
            actual: tensor.shape().to_vec(),
        }
        .into()
    })
}

fn require_boxes_num_tensor(outputs: &TensorMap, name: &str) -> RameResult<Vec<usize>> {
    let tensor = outputs
        .get(name)
        .ok_or_else(|| ModelError::MissingTensor(name.to_string()))?;

    match tensor {
        TensorValue::I32(tensor) => {
            if tensor.shape().len() != 1 {
                return Err(ModelError::InvalidTensorShape {
                    name: name.to_string(),
                    expected: "[batch]".to_string(),
                    actual: tensor.shape().to_vec(),
                }
                .into());
            }

            tensor
                .iter()
                .map(|count| usize::try_from(*count).map_err(|_| invalid_box_count(name, *count)))
                .collect()
        }
        TensorValue::I64(tensor) => {
            if tensor.shape().len() != 1 {
                return Err(ModelError::InvalidTensorShape {
                    name: name.to_string(),
                    expected: "[batch]".to_string(),
                    actual: tensor.shape().to_vec(),
                }
                .into());
            }

            tensor
                .iter()
                .map(|count| usize::try_from(*count).map_err(|_| invalid_box_count(name, *count)))
                .collect()
        }
        tensor => Err(ModelError::InvalidTensorType {
            name: name.to_string(),
            expected: "i32 or i64".to_string(),
            actual: tensor.kind().to_string(),
        }
        .into()),
    }
}

fn invalid_box_count(name: &str, count: impl ToString) -> crate::RameError {
    ModelError::InvalidTensorShape {
        name: name.to_string(),
        expected: format!("non-negative box count, got {}", count.to_string()),
        actual: Vec::new(),
    }
    .into()
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use crate::models::pp_doclayout::boxes::BatchedBoxes;
    use crate::tensor::{TensorMap, TensorValue};

    #[test]
    fn splits_boxes_by_boxes_num() {
        let boxes = Array2::from_shape_vec(
            (3, 6),
            vec![
                0.0, 0.9, 1.0, 2.0, 3.0, 4.0, //
                1.0, 0.8, 5.0, 6.0, 7.0, 8.0, //
                2.0, 0.7, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let outputs = outputs_with_counts(boxes, vec![2, 1]);

        let boxes = BatchedBoxes::from_outputs(&outputs, "boxes", "boxes_num", 6).unwrap();

        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes.item(0).shape(), &[2, 6]);
        assert_eq!(boxes.item(0)[[1, 0]], 1.0);
        assert_eq!(boxes.item(1).shape(), &[1, 6]);
        assert_eq!(boxes.item(1)[[0, 0]], 2.0);
    }

    #[test]
    fn supports_model_specific_box_widths() {
        let boxes =
            Array2::from_shape_vec((1, 7), vec![0.0, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let outputs = outputs_with_counts(boxes, vec![1]);

        let boxes = BatchedBoxes::from_outputs(&outputs, "boxes", "boxes_num", 7).unwrap();

        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes.item(0).shape(), &[1, 7]);
        assert_eq!(boxes.item(0)[[0, 6]], 5.0);
    }

    #[test]
    fn rejects_counts_that_do_not_match_rows() {
        let boxes = Array2::from_shape_vec((1, 6), vec![0.0, 0.9, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let outputs = outputs_with_counts(boxes, vec![2]);

        let err = BatchedBoxes::from_outputs(&outputs, "boxes", "boxes_num", 6).unwrap_err();

        assert!(err.to_string().contains("2 rows from `boxes_num`"));
    }

    #[test]
    fn supports_empty_items() {
        let boxes = Array2::from_shape_vec((1, 6), vec![0.0, 0.9, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let outputs = outputs_with_counts(boxes, vec![0, 1]);

        let boxes = BatchedBoxes::from_outputs(&outputs, "boxes", "boxes_num", 6).unwrap();

        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes.item(0).shape(), &[0, 6]);
        assert_eq!(boxes.item(1).shape(), &[1, 6]);
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
