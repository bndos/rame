use crate::RameResult;
use crate::image::{Image, ImageView};
use crate::layout::LayoutResult;
use crate::runtime::{Decoder, Processor, StandardModelRunner, expect_one};
use crate::session::InferSession;

pub trait LayoutModel {
    fn detect_layout_many_views<'a>(
        &mut self,
        images: &'a [ImageView<'a>],
    ) -> RameResult<Vec<LayoutResult>>;

    fn detect_layout_many(&mut self, images: &[Image]) -> RameResult<Vec<LayoutResult>> {
        let views = images.iter().map(Image::as_view).collect::<Vec<_>>();
        self.detect_layout_many_views(&views)
    }

    fn detect_layout_view<'a>(&mut self, image: ImageView<'a>) -> RameResult<LayoutResult> {
        let results = self.detect_layout_many_views(std::slice::from_ref(&image))?;
        expect_one(results, "layout output")
    }

    fn detect_layout(&mut self, image: &Image) -> RameResult<LayoutResult> {
        self.detect_layout_view(image.as_view())
    }
}

impl<P, S, D> LayoutModel for StandardModelRunner<P, S, D>
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
