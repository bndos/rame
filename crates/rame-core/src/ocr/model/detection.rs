use crate::RameResult;
use crate::image::{Image, ImageView};
use crate::ocr::TextDetectionResult;
use crate::runtime::expect_one;

pub trait TextDetectionModel {
    fn detect_text_many_views<'a>(
        &mut self,
        images: &'a [ImageView<'a>],
    ) -> RameResult<Vec<TextDetectionResult>>;

    fn detect_text_many(&mut self, images: &[Image]) -> RameResult<Vec<TextDetectionResult>> {
        let views = images.iter().map(Image::as_view).collect::<Vec<_>>();
        self.detect_text_many_views(&views)
    }

    fn detect_text_view<'a>(&mut self, image: ImageView<'a>) -> RameResult<TextDetectionResult> {
        let results = self.detect_text_many_views(std::slice::from_ref(&image))?;
        expect_one(results, "text detection output")
    }

    fn detect_text(&mut self, image: &Image) -> RameResult<TextDetectionResult> {
        self.detect_text_view(image.as_view())
    }
}
