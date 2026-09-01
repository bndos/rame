use crate::RameResult;
use crate::image::ImageView;
use crate::layout::{LayoutModel, LayoutResult};
use crate::runtime::{Decoder, ModelArchitecture, ModelBuilder, Processor, StandardModelRunner};
use crate::session::InferSession;

/// PP-DocLayout V3 semantic model.
#[derive(Debug, Clone, Copy)]
pub struct PpDocLayoutV3;

impl PpDocLayoutV3 {
    pub fn builder() -> ModelBuilder<Self> {
        ModelBuilder::new(Self)
    }
}

impl ModelArchitecture for PpDocLayoutV3 {
    type Input<'a> = ImageView<'a>;
    type Output = LayoutResult;
}

impl<P, S, D> LayoutModel for StandardModelRunner<PpDocLayoutV3, P, S, D>
where
    P: for<'a> Processor<Source<'a> = ImageView<'a>>,
    S: InferSession,
    D: Decoder<Output = LayoutResult, Context = P::Context>,
{
    fn detect_layout_many_views<'a>(
        &mut self,
        images: &'a [ImageView<'a>],
    ) -> RameResult<Vec<LayoutResult>> {
        self.run(images)
    }
}
