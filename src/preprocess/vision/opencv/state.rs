use ndarray::{Array2, Array3, Array4};
use opencv::boxed_ref::BoxedRef;
use opencv::core::{Mat, Vec3b};
use opencv::prelude::MatTraitConst;

use crate::RameResult;
use crate::image::Image;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp};
use crate::preprocess::vision::opencv::normalize_permute::NormalizeAndPermute;
use crate::preprocess::vision::{VisionBatchOutput, VisionOp};

#[derive(Debug, Clone, Copy, Default)]
pub struct OpenCvVisionBackend;

#[doc(hidden)]
pub struct OpenCvVisionState<'a> {
    pub(super) image: OpenCvImage<'a>,
    pub(super) source_width: i32,
    pub(super) source_height: i32,
    pub(super) scale_factor: [f32; 2],
}

#[doc(hidden)]
pub struct OpenCvVisionBatch<'a> {
    pub(super) items: Vec<OpenCvVisionState<'a>>,
    pub(super) normalized_images: Option<Vec<Array3<f32>>>,
    pub(super) tensor: Option<Array4<f32>>,
}

#[doc(hidden)]
pub(super) enum OpenCvImage<'a> {
    Borrowed(BoxedRef<'a, Mat>),
    Owned(Mat),
}

impl PreprocessBackend for OpenCvVisionBackend {
    type Source = Image;
    type Batch<'a>
        = OpenCvVisionBatch<'a>
    where
        Self::Source: 'a;
    type Output = VisionBatchOutput;
    type Op = VisionOp;

    fn compile(&self, ops: &mut Vec<Self::Op>) {
        super::compile::compile(ops);
    }

    fn batch<'a>(&self, images: &'a [Self::Source]) -> RameResult<Self::Batch<'a>> {
        OpenCvVisionBatch::new(images)
    }

    fn finish(&self, batch: Self::Batch<'_>) -> RameResult<Self::Output> {
        batch.finish()
    }
}

impl PreprocessOp<OpenCvVisionBackend> for VisionOp {
    fn apply<'a>(&self, batch: &mut OpenCvVisionBatch<'a>) -> RameResult<()>
    where
        Image: 'a,
    {
        match *self {
            Self::Resize(op) => op.apply_opencv(batch),
            Self::NormalizeImage(op) => op.apply_opencv(batch),
            Self::Permute(op) => op.apply_opencv(batch),
            Self::NormalizeAndPermute { normalize, permute } => {
                NormalizeAndPermute::new(normalize, permute).apply_opencv(batch)
            }
        }
    }
}

impl OpenCvImage<'_> {
    pub(super) fn size(&self) -> opencv::Result<opencv::core::Size> {
        match self {
            Self::Borrowed(image) => image.size(),
            Self::Owned(image) => image.size(),
        }
    }
}

impl<'a> OpenCvVisionState<'a> {
    fn new(image: &'a Image) -> RameResult<Self> {
        let source_size = image.size();

        let image = Mat::new_rows_cols_with_bytes::<Vec3b>(
            source_size.height as i32,
            source_size.width as i32,
            image.data(),
        )
        .map(OpenCvImage::Borrowed)
        .map_err(PreprocessError::from)?;

        Ok(Self {
            image,
            source_width: source_size.width as i32,
            source_height: source_size.height as i32,
            scale_factor: [1.0, 1.0],
        })
    }
}

impl<'a> OpenCvVisionBatch<'a> {
    pub(super) fn new(images: &'a [Image]) -> RameResult<Self> {
        let items = images
            .iter()
            .map(OpenCvVisionState::new)
            .collect::<RameResult<Vec<_>>>()?;

        Ok(Self {
            items,
            normalized_images: None,
            tensor: None,
        })
    }

    pub(super) fn finish(self) -> RameResult<VisionBatchOutput> {
        let len = self.items.len();
        let mut image_shapes = Array2::zeros((len, 2));
        let mut scale_factors = Array2::zeros((len, 2));
        let tensor = self.tensor.ok_or(PreprocessError::MissingOutput)?;
        let shape = tensor.shape();

        for (index, state) in self.items.iter().enumerate() {
            image_shapes[[index, 0]] = shape[2] as f32;
            image_shapes[[index, 1]] = shape[3] as f32;
            scale_factors[[index, 0]] = state.scale_factor[0];
            scale_factors[[index, 1]] = state.scale_factor[1];
        }

        Ok(VisionBatchOutput {
            tensor,
            image_shapes,
            scale_factors,
        })
    }
}
