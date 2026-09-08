use numpy::PyReadonlyArray3;
use pyo3::prelude::*;
use rame::layout::LayoutModel;
use rame::models::pp_doclayout::{plus, v3};
use rame::runtime::ModelLoader;
use rame::sources::HuggingFace;

use crate::engine::PyOrtSessionConfig;
use crate::error::into_py_err;
use crate::image::BorrowedImageArray;
use crate::layout::PyLayoutResult;
use rayon::prelude::*;

#[pyclass(name = "PpDocLayoutPlusOnnx")]
pub(crate) struct PyPpDocLayoutPlusOnnx {
    inner: Box<dyn LayoutModel + Send + Sync>,
}

#[pymethods]
impl PyPpDocLayoutPlusOnnx {
    #[new]
    #[pyo3(signature = (source, engine_config=None))]
    fn new(
        py: Python<'_>,
        source: &str,
        engine_config: Option<PyRef<'_, PyOrtSessionConfig>>,
    ) -> PyResult<Self> {
        let hf = HuggingFace::new().map_err(into_py_err)?;
        let hf_source = hf.model(source);
        let session_config = engine_config
            .as_deref()
            .map(PyOrtSessionConfig::to_ort_session_config)
            .transpose()?
            .unwrap_or_default();
        let loader = plus::onnx::Loader::default().session_config(session_config);
        let model = py.detach(|| loader.load(hf_source)).map_err(into_py_err)?;
        Ok(Self {
            inner: Box::new(model),
        })
    }

    #[pyo3(signature = (image, *, copy=true))]
    fn detect_layout(
        &mut self,
        py: Python<'_>,
        image: PyReadonlyArray3<'_, u8>,
        copy: bool,
    ) -> PyResult<PyLayoutResult> {
        let image = BorrowedImageArray::new(&image)?;
        if copy {
            let image = image.to_owned_image()?;
            py.detach(|| self.inner.detect_layout(&image))
                .map(PyLayoutResult::from)
                .map_err(into_py_err)
        } else {
            let image = image.as_image_view()?;
            self.inner
                .detect_layout_view(image)
                .map(PyLayoutResult::from)
                .map_err(into_py_err)
        }
    }

    #[pyo3(signature = (images, *, copy=true))]
    fn detect_layout_many(
        &mut self,
        py: Python<'_>,
        images: Vec<PyReadonlyArray3<'_, u8>>,
        copy: bool,
    ) -> PyResult<Vec<PyLayoutResult>> {
        let image_arrays: Vec<_> = images
            .iter()
            .map(BorrowedImageArray::new)
            .collect::<PyResult<_>>()?;
        if copy {
            let images: Vec<_> = image_arrays
                .par_iter()
                .map(|image| image.to_owned_image())
                .collect::<PyResult<_>>()?;
            py.detach(|| self.inner.detect_layout_many(&images))
                .map(|rs| rs.into_iter().map(PyLayoutResult::from).collect())
                .map_err(into_py_err)
        } else {
            let images: Vec<_> = image_arrays
                .iter()
                .map(|image| image.as_image_view())
                .collect::<PyResult<_>>()?;
            self.inner
                .detect_layout_many_views(&images)
                .map(|rs| rs.into_iter().map(PyLayoutResult::from).collect())
                .map_err(into_py_err)
        }
    }
}

#[pyclass(name = "PpDocLayoutV3Onnx")]
pub(crate) struct PyPpDocLayoutV3Onnx {
    inner: Box<dyn LayoutModel + Send + Sync>,
}

#[pymethods]
impl PyPpDocLayoutV3Onnx {
    #[new]
    #[pyo3(signature = (source, engine_config=None))]
    fn new(
        py: Python<'_>,
        source: &str,
        engine_config: Option<PyRef<'_, PyOrtSessionConfig>>,
    ) -> PyResult<Self> {
        let hf = HuggingFace::new().map_err(into_py_err)?;
        let hf_source = hf.model(source);
        let session_config = engine_config
            .as_deref()
            .map(PyOrtSessionConfig::to_ort_session_config)
            .transpose()?
            .unwrap_or_default();
        let loader = v3::onnx::Loader::default().session_config(session_config);
        let model = py.detach(|| loader.load(hf_source)).map_err(into_py_err)?;
        Ok(Self {
            inner: Box::new(model),
        })
    }

    #[pyo3(signature = (image, *, copy=true))]
    fn detect_layout(
        &mut self,
        py: Python<'_>,
        image: PyReadonlyArray3<'_, u8>,
        copy: bool,
    ) -> PyResult<PyLayoutResult> {
        let image = BorrowedImageArray::new(&image)?;
        if copy {
            let image = image.to_owned_image()?;
            py.detach(|| self.inner.detect_layout(&image))
                .map(PyLayoutResult::from)
                .map_err(into_py_err)
        } else {
            let image = image.as_image_view()?;
            self.inner
                .detect_layout_view(image)
                .map(PyLayoutResult::from)
                .map_err(into_py_err)
        }
    }

    #[pyo3(signature = (images, *, copy=true))]
    fn detect_layout_many(
        &mut self,
        py: Python<'_>,
        images: Vec<PyReadonlyArray3<'_, u8>>,
        copy: bool,
    ) -> PyResult<Vec<PyLayoutResult>> {
        let image_arrays: Vec<_> = images
            .iter()
            .map(BorrowedImageArray::new)
            .collect::<PyResult<_>>()?;
        if copy {
            let images: Vec<_> = image_arrays
                .par_iter()
                .map(|image| image.to_owned_image())
                .collect::<PyResult<_>>()?;
            py.detach(|| self.inner.detect_layout_many(&images))
                .map(|rs| rs.into_iter().map(PyLayoutResult::from).collect())
                .map_err(into_py_err)
        } else {
            let images: Vec<_> = image_arrays
                .iter()
                .map(|image| image.as_image_view())
                .collect::<PyResult<_>>()?;
            self.inner
                .detect_layout_many_views(&images)
                .map(|rs| rs.into_iter().map(PyLayoutResult::from).collect())
                .map_err(into_py_err)
        }
    }
}
