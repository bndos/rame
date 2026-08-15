use opencv::boxed_ref::BoxedRef;
use opencv::core::{Mat, Vec3b};
use opencv::prelude::MatTraitConst;

use crate::RameResult;
use crate::image::ImageView;
use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp};
use crate::preprocess::vision::VisionBatchOutput;
use crate::preprocess::{PreprocessConfig, PreprocessError};
use crate::tensor::Device;

#[derive(Debug, Clone)]
pub struct OpenCvVisionBackend {
    device: Device,
}

impl Default for OpenCvVisionBackend {
    fn default() -> Self {
        Self::new(Device::cpu())
    }
}

impl OpenCvVisionBackend {
    pub fn new(device: impl Into<Device>) -> Self {
        Self {
            device: device.into(),
        }
    }
}

#[doc(hidden)]
pub struct OpenCvVisionBatch<'a> {
    pub(super) images: Vec<OpenCvImage<'a>>,
    pub(super) source_sizes: Vec<[i32; 2]>,
    pub(super) scale_factors: Vec<[f32; 2]>,
    pub(super) device: Device,
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
        OpenCvVisionBatch::new(images, self.device.clone()).map(OpenCvVisionData::ImageBatch)
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

impl OpenCvImage<'_> {
    fn borrowed<'a>(image: &'a ImageView<'a>) -> RameResult<OpenCvImage<'a>> {
        let source_size = image.size();

        Ok(Mat::new_rows_cols_with_bytes::<Vec3b>(
            source_size.height as i32,
            source_size.width as i32,
            image.data(),
        )
        .map(OpenCvImage::Borrowed)
        .map_err(PreprocessError::from)?)
    }
}

impl<'a> OpenCvVisionBatch<'a> {
    pub(super) fn new(images: &'a [ImageView<'a>], device: Device) -> RameResult<Self> {
        let mut batch_images = Vec::with_capacity(images.len());
        let mut source_sizes = Vec::with_capacity(images.len());
        let mut scale_factors = Vec::with_capacity(images.len());

        for image in images {
            let size = image.size();
            source_sizes.push([size.height as i32, size.width as i32]);
            batch_images.push(OpenCvImage::borrowed(image)?);
            scale_factors.push([1.0, 1.0]);
        }

        Ok(Self {
            images: batch_images,
            source_sizes,
            scale_factors,
            device,
        })
    }
}

impl From<PreprocessConfig> for OpenCvVisionBackend {
    fn from(config: PreprocessConfig) -> Self {
        Self::new(config.device)
    }
}
