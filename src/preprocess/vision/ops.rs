/// Interpolation algorithm used for image resizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Cubic,
}

/// Tensor layout produced by layout conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    /// Batch, channel, height, width.
    Nchw,
}

/// How a resize operation determines output dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResizeMode {
    FixedSize {
        width: usize,
        height: usize,
    },
    Scale {
        /// Scale applied to the width dimension.
        scale_width: f32,
        /// Scale applied to the height dimension.
        scale_height: f32,
    },
}

/// Resize operation configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resize {
    pub mode: ResizeMode,
    pub interpolation: Interpolation,
}

impl Resize {
    pub fn fixed_size(width: usize, height: usize, interpolation: Interpolation) -> Self {
        Self {
            mode: ResizeMode::FixedSize { width, height },
            interpolation,
        }
    }

    pub fn fixed_square(size: usize, interpolation: Interpolation) -> Self {
        Self::fixed_size(size, size, interpolation)
    }

    pub fn scale(scale_width: f32, scale_height: f32, interpolation: Interpolation) -> Self {
        Self {
            mode: ResizeMode::Scale {
                scale_width,
                scale_height,
            },
            interpolation,
        }
    }
}

/// Applies model-provided image normalization constants.
///
/// Backends apply this as `(pixel * scale - mean) / std` and produce f32 image
/// data for later tensor layout conversion. The constants belong to the model
/// artifact, not to the backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizeImage {
    /// Multiplies raw byte channels before mean/std normalization.
    pub scale: f32,
    /// Per-channel constants from the model's preprocessing metadata.
    pub mean: [f32; 3],
    /// Per-channel constants from the model's preprocessing metadata.
    pub std: [f32; 3],
}

impl NormalizeImage {
    pub const INV_255: f32 = 1.0 / 255.0;

    pub fn new(scale: f32, mean: [f32; 3], std: [f32; 3]) -> Self {
        Self { scale, mean, std }
    }

    pub fn scale(scale: f32) -> Self {
        Self::new(scale, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    }
}

/// Tensor conversion operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToTensor {
    pub layout: TensorLayout,
    pub normalize: Option<NormalizeImage>,
}

impl ToTensor {
    pub fn new(layout: TensorLayout) -> Self {
        Self {
            layout,
            normalize: None,
        }
    }

    pub fn nchw() -> Self {
        Self::new(TensorLayout::Nchw)
    }

    pub fn normalize(mut self, normalize: NormalizeImage) -> Self {
        self.normalize = Some(normalize);
        self
    }
}

/// Typed vision preprocessing operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VisionOp {
    Resize(Resize),
    ToTensor(ToTensor),
}

impl From<Resize> for VisionOp {
    fn from(op: Resize) -> Self {
        Self::Resize(op)
    }
}

impl From<ToTensor> for VisionOp {
    fn from(op: ToTensor) -> Self {
        Self::ToTensor(op)
    }
}
