use opencv::core::{self, Mat, Size, ToInputArray, Vector};
use opencv::prelude::MatTraitConst;

use crate::RameResult;
use crate::preprocess::PreprocessError;
use crate::preprocess::vision::opencv::state::{OpenCvImage, OpenCvVisionBatch};
use crate::preprocess::vision::{NormalizeImage, TensorLayout, ToTensor, VisionBatchOutput};
use crate::tensor::Tensor;

pub(in crate::preprocess::vision::opencv) fn to_tensor(
    op: &ToTensor,
    batch: OpenCvVisionBatch<'_>,
) -> RameResult<VisionBatchOutput> {
    match op.layout {
        TensorLayout::Nchw => apply_nchw(op, batch),
    }
}

fn output(
    data: Vec<f32>,
    shape: [usize; 4],
    image_shapes: Vec<f32>,
    scale_factors: Vec<f32>,
) -> RameResult<VisionBatchOutput> {
    let tensor = Tensor::from_vec(data, &shape).map_err(|err| PreprocessError::Backend {
        backend: "candle",
        message: err.to_string(),
    })?;
    let len = shape[0];

    let image_shapes = metadata_tensor(image_shapes, len)?;
    let scale_factors = metadata_tensor(scale_factors, len)?;

    Ok(VisionBatchOutput {
        tensor,
        image_shapes,
        scale_factors,
    })
}

fn apply_nchw(op: &ToTensor, batch: OpenCvVisionBatch<'_>) -> RameResult<VisionBatchOutput> {
    if batch.images.is_empty() {
        return output(Vec::new(), [0, 3, 0, 0], Vec::new(), Vec::new());
    }

    let size = batch.images[0].size().map_err(PreprocessError::from)?;
    let (height, width) = (size.height as usize, size.width as usize);
    let len = batch.images.len();
    let mut tensor_data = vec![0.0; len * 3 * height * width];
    let mut image_shapes = vec![0.0; len * 2];
    let mut scale_factors = vec![0.0; len * 2];
    let plane = height * width;

    for (index, (output, image)) in tensor_data
        .chunks_mut(3 * plane)
        .zip(batch.images.iter())
        .enumerate()
    {
        ensure_size(image, size)?;
        image_into_nchw(op, image, height, width, plane, output)?;

        let metadata_offset = index * 2;
        image_shapes[metadata_offset] = height as f32;
        image_shapes[metadata_offset + 1] = width as f32;
        scale_factors[metadata_offset] = batch.scale_factors[index][0];
        scale_factors[metadata_offset + 1] = batch.scale_factors[index][1];
    }

    output(
        tensor_data,
        [len, 3, height, width],
        image_shapes,
        scale_factors,
    )
}

fn image_into_nchw(
    op: &ToTensor,
    image: &OpenCvImage<'_>,
    height: usize,
    width: usize,
    plane: usize,
    output: &mut [f32],
) -> RameResult<()> {
    match image {
        OpenCvImage::Borrowed(image) => mat_into_nchw(op, image, height, width, plane, output),
        OpenCvImage::Owned(image) => mat_into_nchw(op, image, height, width, plane, output),
    }
}

fn mat_into_nchw(
    op: &ToTensor,
    image: &impl ToInputArray,
    height: usize,
    width: usize,
    plane: usize,
    output: &mut [f32],
) -> RameResult<()> {
    let mut channels = Vector::<Mat>::new();
    let coefficients = NormalizationCoefficients::from(op.normalize);
    core::split(image, &mut channels).map_err(PreprocessError::from)?;

    for channel in 0..3 {
        let source = channels.get(channel).map_err(PreprocessError::from)?;
        let start = channel * plane;
        let end = start + plane;
        // OpenCV writes into this Mat header, which views the corresponding
        // channel plane inside the output tensor buffer.
        let mut target =
            Mat::new_rows_cols_with_data_mut(height as i32, width as i32, &mut output[start..end])
                .map_err(PreprocessError::from)?;

        source
            .convert_to(
                &mut target,
                core::CV_32FC1,
                coefficients.scale[channel],
                coefficients.bias[channel],
            )
            .map_err(PreprocessError::from)?;
    }

    Ok(())
}

fn metadata_tensor(data: Vec<f32>, len: usize) -> RameResult<Tensor> {
    Tensor::from_vec(data, &[len, 2]).map_err(|err| {
        PreprocessError::Backend {
            backend: "candle",
            message: err.to_string(),
        }
        .into()
    })
}

#[derive(Debug, Clone, Copy)]
struct NormalizationCoefficients {
    scale: [f64; 3],
    bias: [f64; 3],
}

impl From<Option<NormalizeImage>> for NormalizationCoefficients {
    fn from(normalize: Option<NormalizeImage>) -> Self {
        let Some(normalize) = normalize else {
            return Self {
                scale: [1.0, 1.0, 1.0],
                bias: [0.0, 0.0, 0.0],
            };
        };

        Self {
            scale: [
                (normalize.scale / normalize.std[0]) as f64,
                (normalize.scale / normalize.std[1]) as f64,
                (normalize.scale / normalize.std[2]) as f64,
            ],
            bias: [
                (-normalize.mean[0] / normalize.std[0]) as f64,
                (-normalize.mean[1] / normalize.std[1]) as f64,
                (-normalize.mean[2] / normalize.std[2]) as f64,
            ],
        }
    }
}

fn ensure_size(image: &OpenCvImage<'_>, expected: Size) -> RameResult<()> {
    let actual = image.size().map_err(PreprocessError::from)?;
    if actual == expected {
        return Ok(());
    }

    Err(PreprocessError::InvalidTensorShape {
        name: "image",
        expected: format!("[height={}, width={}]", expected.height, expected.width),
        actual: vec![actual.height as usize, actual.width as usize],
    }
    .into())
}
