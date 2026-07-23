/// Interpolation algorithm used for image resizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Cubic,
}

/// Tensor layout produced by image-to-tensor preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    /// Batch, channel, height, width.
    Nchw,
}

/// Scalar conversion applied when turning image bytes into tensor values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelScale {
    /// Converts `[0, 255]` bytes into `[0.0, 1.0]`.
    OneOver255,
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

/// Image normalization and tensor layout conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizeAndPermute {
    pub scale: PixelScale,
    pub layout: TensorLayout,
}

impl NormalizeAndPermute {
    pub fn new(scale: PixelScale, layout: TensorLayout) -> Self {
        Self { scale, layout }
    }
}
