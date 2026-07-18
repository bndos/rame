use crate::RameResult;

/// Image dimensions in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl Size {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// Pixel storage format for an image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb8,
}

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("expected {expected_len} bytes for {width}x{height} RGB image, got {actual_len}")]
    InvalidRgbData {
        width: u32,
        height: u32,
        expected_len: usize,
        actual_len: usize,
    },
}

/// In-memory image input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    size: Size,
    pixel_format: PixelFormat,
    data: Vec<u8>,
}

impl Image {
    pub fn from_rgb8(width: u32, height: u32, data: Vec<u8>) -> RameResult<Self> {
        let expected_len = width as usize * height as usize * 3;
        if data.len() != expected_len {
            return Err(ImageError::InvalidRgbData {
                width,
                height,
                expected_len,
                actual_len: data.len(),
            }
            .into());
        }

        Ok(Self {
            size: Size::new(width, height),
            pixel_format: PixelFormat::Rgb8,
            data,
        })
    }

    pub fn size(&self) -> Size {
        self.size
    }

    pub fn pixel_format(&self) -> &PixelFormat {
        &self.pixel_format
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }
}
