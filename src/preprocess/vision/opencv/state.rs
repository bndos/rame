use opencv::boxed_ref::BoxedRef;
use opencv::core::{Mat, Vec3b};
use opencv::prelude::MatTraitConst;

use crate::RameResult;
use crate::image::ImageView;
use crate::preprocess::PreprocessError;
use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp};
use crate::preprocess::vision::VisionBatchOutput;

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
}

#[doc(hidden)]
pub enum OpenCvVisionData<'a> {
    ImageBatch(OpenCvVisionBatch<'a>),
    TensorBatch(VisionBatchOutput),
}

#[doc(hidden)]
pub(super) enum OpenCvImage<'a> {
    Borrowed(BoxedRef<'a, Mat>),
    Owned(Mat),
}

impl PreprocessBackend for OpenCvVisionBackend {
    type Source<'a> = ImageView<'a>;
    type Data<'a> = OpenCvVisionData<'a>;
    type Output = VisionBatchOutput;

    fn input<'a>(&self, images: &'a [Self::Source<'a>]) -> RameResult<Self::Data<'a>> {
        OpenCvVisionBatch::new(images).map(OpenCvVisionData::ImageBatch)
    }

    fn finish(&self, data: Self::Data<'_>) -> RameResult<Self::Output> {
        match data {
            OpenCvVisionData::ImageBatch(_) => Err(PreprocessError::MissingOutput.into()),
            OpenCvVisionData::TensorBatch(output) => Ok(output),
        }
    }

    fn compile(&self, ops: Vec<Box<dyn PreprocessOp<Self>>>) -> Vec<Box<dyn PreprocessOp<Self>>> {
        ops
    }
}

impl<'a> OpenCvVisionData<'a> {
    pub(super) fn into_image_batch(self) -> RameResult<OpenCvVisionBatch<'a>> {
        match self {
            Self::ImageBatch(batch) => Ok(batch),
            Self::TensorBatch(_) => Err(PreprocessError::InvalidTensorShape {
                name: "opencv preprocess data",
                expected: "image batch".to_string(),
                actual: vec![],
            }
            .into()),
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
    fn new(image: &'a ImageView<'a>) -> RameResult<Self> {
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
    pub(super) fn new(images: &'a [ImageView<'a>]) -> RameResult<Self> {
        let items = images
            .iter()
            .map(OpenCvVisionState::new)
            .collect::<RameResult<Vec<_>>>()?;

        Ok(Self { items })
    }
}
