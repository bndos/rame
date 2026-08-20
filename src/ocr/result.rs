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

/// Result of text recognition on one cropped textline image.
#[derive(Debug, Clone, PartialEq)]
pub struct TextRecognitionResult {
    pub recognition: TextRecognition,
}

impl TextRecognitionResult {
    pub fn new(recognition: TextRecognition) -> Self {
        Self { recognition }
    }
}

/// Recognized text for one cropped textline image.
#[derive(Debug, Clone, PartialEq)]
pub struct TextRecognition {
    pub text: String,
    pub score: f32,
}
