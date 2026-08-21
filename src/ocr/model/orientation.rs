use crate::RameResult;
use crate::image::{Image, ImageView};
use crate::ocr::TextLineOrientationResult;
use crate::runtime::expect_one;

pub trait TextLineOrientationModel {
    fn classify_textline_orientation_many_views<'a>(
        &mut self,
        images: &'a [ImageView<'a>],
    ) -> RameResult<Vec<TextLineOrientationResult>>;

    fn classify_textline_orientation_many(
        &mut self,
        images: &[Image],
    ) -> RameResult<Vec<TextLineOrientationResult>> {
        let views = images.iter().map(Image::as_view).collect::<Vec<_>>();
        self.classify_textline_orientation_many_views(&views)
    }

    fn classify_textline_orientation_view<'a>(
        &mut self,
        image: ImageView<'a>,
    ) -> RameResult<TextLineOrientationResult> {
        let results =
            self.classify_textline_orientation_many_views(std::slice::from_ref(&image))?;
        expect_one(results, "textline orientation output")
    }

    fn classify_textline_orientation(
        &mut self,
        image: &Image,
    ) -> RameResult<TextLineOrientationResult> {
        self.classify_textline_orientation_view(image.as_view())
    }
}
