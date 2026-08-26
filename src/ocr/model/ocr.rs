use crate::RameResult;
use crate::image::{Image, ImageView};
use crate::ocr::OcrResult;
use crate::runtime::expect_one;

pub trait OcrModel {
    fn recognize_many_views<'a>(
        &mut self,
        images: &'a [ImageView<'a>],
    ) -> RameResult<Vec<OcrResult>>;

    fn recognize_many(&mut self, images: &[Image]) -> RameResult<Vec<OcrResult>> {
        let views = images.iter().map(Image::as_view).collect::<Vec<_>>();
        self.recognize_many_views(&views)
    }

    fn recognize_view<'a>(&mut self, image: ImageView<'a>) -> RameResult<OcrResult> {
        let results = self.recognize_many_views(std::slice::from_ref(&image))?;
        expect_one(results, "OCR output")
    }

    fn recognize(&mut self, image: &Image) -> RameResult<OcrResult> {
        self.recognize_view(image.as_view())
    }
}
