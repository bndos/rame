mod model;
mod result;

pub use model::{TextDetectionModel, TextRecognitionModel};
pub use result::{
    OcrLine, OcrResult, TextDetection, TextDetectionResult, TextRecognition, TextRecognitionResult,
};
