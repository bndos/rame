mod model;
mod result;

pub use model::{TextDetectionModel, TextLineOrientationModel, TextRecognitionModel};
pub use result::{
    OcrLine, OcrResult, TextDetection, TextDetectionResult, TextLineOrientation,
    TextLineOrientationResult, TextRecognitionResult,
};
