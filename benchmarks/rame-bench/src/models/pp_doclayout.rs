use crate::datasets::ImageSample;
use crate::error::BenchResult;
use crate::models::LayoutModel;

#[derive(Debug, Default)]
pub struct PpDocLayoutPlusOnnx;

impl PpDocLayoutPlusOnnx {
    pub fn new() -> Self {
        Self
    }
}

impl LayoutModel for PpDocLayoutPlusOnnx {
    fn predict_many(&mut self, samples: &[ImageSample]) -> BenchResult<()> {
        let _ = samples;
        Ok(())
    }
}
