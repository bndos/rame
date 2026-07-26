use crate::RameResult;
use crate::image::Image;
use crate::layout::LayoutResult;

pub trait LayoutModel {
    fn detect_layout_many(&mut self, images: &[Image]) -> RameResult<Vec<LayoutResult>>;

    fn detect_layout(&mut self, image: &Image) -> RameResult<LayoutResult> {
        let results = self.detect_layout_many(std::slice::from_ref(image))?;
        let [result]: [_; 1] = results.try_into().map_err(|results: Vec<LayoutResult>| {
            crate::RameError::InvalidBatchLength {
                stage: "layout output",
                expected: 1,
                actual: results.len(),
            }
        })?;

        Ok(result)
    }
}
