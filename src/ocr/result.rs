use crate::geometry::{Polygon, Rect};

/// Result of a full OCR run on one image.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrResult {
    pub lines: Vec<OcrLine>,
}

impl OcrResult {
    pub fn new(lines: Vec<OcrLine>) -> Self {
        Self { lines }
    }
}

/// Recognized text associated with one region in the source image.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrLine {
    pub text: String,
    pub polygon: Polygon,
    pub rect: Option<Rect>,
    pub score: f32,
}

/// Result of text detection on one image.
#[derive(Debug, Clone, PartialEq)]
pub struct TextDetectionResult {
    pub detections: Vec<TextDetection>,
}

impl TextDetectionResult {
    pub fn new(detections: Vec<TextDetection>) -> Self {
        Self { detections }
    }
}

/// One detected text region.
#[derive(Debug, Clone, PartialEq)]
pub struct TextDetection {
    pub polygon: Polygon,
    pub score: f32,
}

/// Recognized text for one cropped textline image.
#[derive(Debug, Clone, PartialEq)]
pub struct TextRecognitionResult {
    pub text: String,
    pub score: f32,
}

impl TextRecognitionResult {
    pub fn new(text: impl Into<String>, score: f32) -> Self {
        Self {
            text: text.into(),
            score,
        }
    }
}

/// Result of textline orientation classification on one cropped textline image.
#[derive(Debug, Clone, PartialEq)]
pub struct TextLineOrientationResult {
    pub orientation: TextLineOrientation,
    pub score: f32,
}

impl TextLineOrientationResult {
    pub fn new(orientation: TextLineOrientation, score: f32) -> Self {
        Self { orientation, score }
    }
}

/// Orientation prediction for a cropped textline image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextLineOrientation {
    Deg0,
    Deg180,
    Unknown(i32),
}
