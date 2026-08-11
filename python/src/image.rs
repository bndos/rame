use numpy::{PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rame::image::{Image, ImageView};

pub(crate) struct BorrowedImageArray<'a> {
    width: u32,
    height: u32,
    data: &'a [u8],
}

impl<'a> BorrowedImageArray<'a> {
    pub(crate) fn new(array: &'a PyReadonlyArray3<'_, u8>) -> PyResult<Self> {
        let shape = array.shape();
        let (height, width, channels) = (shape[0], shape[1], shape[2]);
        if channels != 3 {
            return Err(PyValueError::new_err(format!(
                "expected 3 channels (RGB), got {channels}"
            )));
        }

        Ok(Self {
            width: width as u32,
            height: height as u32,
            data: array.as_slice()?,
        })
    }

    pub(crate) fn to_owned_image(&self) -> PyResult<Image> {
        Image::from_rgb8(self.width, self.height, self.data).map_err(crate::error::into_py_err)
    }

    pub(crate) fn as_image_view(&self) -> PyResult<ImageView<'_>> {
        ImageView::from_rgb8(self.width, self.height, self.data).map_err(crate::error::into_py_err)
    }
}
