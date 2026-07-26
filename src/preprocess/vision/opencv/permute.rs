use ndarray::Axis;

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::PreprocessOp;
use crate::preprocess::vision::opencv::state::{OpenCvVisionBackend, OpenCvVisionBatch};
use crate::preprocess::vision::{Permute, TensorLayout};

impl PreprocessOp<OpenCvVisionBackend> for Permute {
    fn apply(&self, batch: &mut OpenCvVisionBatch) -> RameResult<()> {
        for item in &mut batch.items {
            let image = item
                .normalized_image
                .as_ref()
                .ok_or(PreprocessError::MissingOutput)?;
            item.tensor = Some(match self.layout {
                TensorLayout::Nchw => image
                    .view()
                    .permuted_axes([2, 0, 1])
                    .insert_axis(Axis(0))
                    .to_owned(),
            });
        }

        Ok(())
    }
}
