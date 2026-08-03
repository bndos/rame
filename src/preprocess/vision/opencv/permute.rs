use ndarray::{Array4, Axis, s};

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::vision::opencv::state::OpenCvVisionBatch;
use crate::preprocess::vision::{Permute, TensorLayout};

impl Permute {
    pub(super) fn apply_opencv(&self, batch: &mut OpenCvVisionBatch) -> RameResult<()> {
        let images = batch
            .normalized_images
            .take()
            .ok_or(PreprocessError::MissingOutput)?;

        batch.tensor = Some(match self.layout {
            TensorLayout::Nchw => nchw_tensor_from_images(images)?,
        });

        Ok(())
    }
}

fn nchw_tensor_from_images(images: Vec<ndarray::Array3<f32>>) -> RameResult<Array4<f32>> {
    let Some(first) = images.first() else {
        return Err(PreprocessError::MissingOutput.into());
    };
    let shape = first.shape();
    let height = shape[0];
    let width = shape[1];
    let channels = shape[2];
    let mut tensor = Array4::zeros((images.len(), channels, height, width));

    for (index, image) in images.into_iter().enumerate() {
        let shape = image.shape();
        if shape != [height, width, channels] {
            return Err(PreprocessError::InvalidTensorShape {
                name: "image",
                expected: format!("[{height}, {width}, {channels}]"),
                actual: shape.to_vec(),
            }
            .into());
        }

        tensor.slice_mut(s![index, .., .., ..]).assign(
            &image
                .view()
                .permuted_axes([2, 0, 1])
                .insert_axis(Axis(0))
                .index_axis(Axis(0), 0),
        );
    }

    Ok(tensor)
}
