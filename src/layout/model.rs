use crate::RameResult;
use crate::image::Image;
use crate::layout::LayoutResult;

pub trait LayoutModel {
    fn detect_layout(&mut self, image: &Image) -> RameResult<LayoutResult>;
}
