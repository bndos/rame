use numpy::{PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rame::image::ImageView;

pub(crate) fn array_to_image_view<'a>(
    array: &'a PyReadonlyArray3<'_, u8>,
) -> PyResult<ImageView<'a>> {
    let shape = array.shape();
    let (height, width, channels) = (shape[0], shape[1], shape[2]);
    if channels != 3 {
        return Err(PyValueError::new_err(format!(
            "expected 3 channels (RGB), got {channels}"
        )));
    }
    ImageView::from_rgb8(width as u32, height as u32, array.as_slice()?)
        .map_err(crate::error::into_py_err)
}
