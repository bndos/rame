use crate::error::BenchResult;
use crate::models::LayoutModel;
use rame::image::Image;
use rame::layout::LayoutModel as RameLayoutModel;
use rame::models::pp_doclayout::plus::{self, PpDocLayoutPlus};
use rame::sources::HuggingFace;

const SOURCE: &str = "PaddlePaddle/PP-DocLayout_plus-L_onnx";

pub struct PpDocLayoutPlusOnnx {
    model: Box<dyn RameLayoutModel>,
}

impl PpDocLayoutPlusOnnx {
    pub fn new() -> BenchResult<Self> {
        let source = HuggingFace::new()?.model(SOURCE);
        let model = PpDocLayoutPlus::builder()
            .source(source)
            .artifact(plus::onnx::Artifact::default())
            .build()?;

        Ok(Self {
            model: Box::new(model),
        })
    }
}

impl LayoutModel for PpDocLayoutPlusOnnx {
    fn predict_many(&mut self, images: &[Image]) -> BenchResult<()> {
        self.model.detect_layout_many(images)?;
        Ok(())
    }
}
