use crate::RameResult;
use crate::image::Image;
use crate::layout::{LayoutModel, LayoutResult};
use crate::runtime::{Decoder, InferencePipeline, ModelArchitecture, ModelBuilder, Processor};
use crate::session::InferSession;

/// PP-DocLayout Plus semantic model.
#[derive(Debug, Clone, Copy)]
pub struct PpDocLayoutPlus;

impl PpDocLayoutPlus {
    pub fn builder() -> ModelBuilder<Self> {
        ModelBuilder::new(Self)
    }
}

impl ModelArchitecture for PpDocLayoutPlus {
    type Output = LayoutResult;
}

impl<P, S, D> LayoutModel for InferencePipeline<PpDocLayoutPlus, P, S, D>
where
    P: Processor<Source = Image>,
    S: InferSession,
    D: Decoder<Output = LayoutResult, Context = P::Context>,
{
    fn detect_layout(&mut self, image: &Image) -> RameResult<LayoutResult> {
        self.run(image)
    }
}
