use numpy::{PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rame::image::Image;

pub(crate) fn array_to_image(array: PyReadonlyArray3<'_, u8>) -> PyResult<Image> {
    let shape = array.shape();
    let (height, width, channels) = (shape[0], shape[1], shape[2]);
    if channels != 3 {
        return Err(PyValueError::new_err(format!(
            "expected 3 channels (RGB), got {channels}"
        )));
    }
    Image::from_rgb8(width as u32, height as u32, array.as_slice()?).map_err(crate::error::into_py_err)
}
