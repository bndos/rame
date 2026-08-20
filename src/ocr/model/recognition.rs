use crate::RameResult;
use crate::image::{Image, ImageView};
use crate::ocr::TextRecognitionResult;
use crate::runtime::expect_one;

pub trait TextRecognitionModel {
    fn recognize_text_many_views<'a>(
        &mut self,
        images: &'a [ImageView<'a>],
    ) -> RameResult<Vec<TextRecognitionResult>>;

    fn recognize_text_many(&mut self, images: &[Image]) -> RameResult<Vec<TextRecognitionResult>> {
        let views = images.iter().map(Image::as_view).collect::<Vec<_>>();
        self.recognize_text_many_views(&views)
    }

    fn recognize_text_view<'a>(
        &mut self,
        image: ImageView<'a>,
    ) -> RameResult<TextRecognitionResult> {
        let results = self.recognize_text_many_views(std::slice::from_ref(&image))?;
        expect_one(results, "text recognition output")
    }

    fn recognize_text(&mut self, image: &Image) -> RameResult<TextRecognitionResult> {
        self.recognize_text_view(image.as_view())
    }
}
