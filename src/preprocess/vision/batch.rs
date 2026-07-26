use ndarray::{Array4, Axis, s};

use crate::RameResult;
use crate::preprocess::PreprocessError;

#[derive(Debug)]
pub struct NchwBatchBuilder {
    len: usize,
    next: usize,
    tensor: Option<Array4<f32>>,
}

impl NchwBatchBuilder {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            next: 0,
            tensor: None,
        }
    }

    pub fn push(&mut self, tensor: Array4<f32>) -> RameResult<usize> {
        self.ensure_batch(&tensor)?;
        let batch = self
            .tensor
            .as_mut()
            .expect("NCHW batch is initialized before assignment");
        let index = self.next;

        batch
            .slice_mut(s![index, .., .., ..])
            .assign(&tensor.index_axis(Axis(0), 0));
        self.next += 1;

        Ok(index)
    }

    pub fn finish(self) -> RameResult<Array4<f32>> {
        if self.next != self.len {
            return Err(PreprocessError::InvalidTensorShape {
                name: "image",
                expected: format!("{} batched images", self.len),
                actual: vec![self.next],
            }
            .into());
        }

        self.tensor
            .ok_or(PreprocessError::MissingOutput)
            .map_err(Into::into)
    }

    fn ensure_batch(&mut self, tensor: &Array4<f32>) -> RameResult<()> {
        let shape = tensor.shape();

        if let Some(batch) = &self.tensor {
            let expected_shape = [1, batch.shape()[1], batch.shape()[2], batch.shape()[3]];
            if shape != expected_shape {
                return Err(PreprocessError::InvalidTensorShape {
                    name: "image",
                    expected: format!("{expected_shape:?}"),
                    actual: shape.to_vec(),
                }
                .into());
            }
            return Ok(());
        }

        self.tensor = Some(Array4::zeros((self.len, shape[1], shape[2], shape[3])));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array4;

    use crate::preprocess::vision::NchwBatchBuilder;

    #[test]
    fn builds_nchw_batch_one_item_at_a_time() {
        let mut batch = NchwBatchBuilder::new(2);

        assert_eq!(batch.push(Array4::from_elem((1, 3, 2, 2), 1.0)).unwrap(), 0);
        assert_eq!(batch.push(Array4::from_elem((1, 3, 2, 2), 2.0)).unwrap(), 1);

        let tensor = batch.finish().unwrap();

        assert_eq!(tensor.shape(), &[2, 3, 2, 2]);
        assert_eq!(tensor[[0, 0, 0, 0]], 1.0);
        assert_eq!(tensor[[1, 0, 0, 0]], 2.0);
    }

    #[test]
    fn rejects_mismatched_item_shapes() {
        let mut batch = NchwBatchBuilder::new(2);
        batch.push(Array4::from_elem((1, 3, 2, 2), 1.0)).unwrap();

        let err = batch
            .push(Array4::from_elem((1, 3, 4, 2), 2.0))
            .unwrap_err();

        assert!(err.to_string().contains("expected [1, 3, 2, 2]"));
    }
}
