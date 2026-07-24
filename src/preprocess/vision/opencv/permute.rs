use ndarray::{Array3, Array4, Axis};

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::PreprocessOp;
use crate::preprocess::vision::opencv::state::{OpenCvVisionBackend, OpenCvVisionState};
use crate::preprocess::vision::{Permute, TensorLayout};

impl PreprocessOp<OpenCvVisionBackend> for Permute {
    fn apply(&self, state: &mut OpenCvVisionState) -> RameResult<()> {
        let image = state
            .normalized_image
            .as_ref()
            .ok_or(PreprocessError::MissingOutput)?;
        state.tensor = Some(permute_image(image, self.layout));
        Ok(())
    }
}

fn permute_image(image: &Array3<f32>, layout: TensorLayout) -> Array4<f32> {
    match layout {
        TensorLayout::Nchw => image
            .view()
            .permuted_axes([2, 0, 1])
            .insert_axis(Axis(0))
            .to_owned(),
    }
}
