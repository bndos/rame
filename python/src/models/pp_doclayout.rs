use numpy::PyReadonlyArray3;
use pyo3::prelude::*;
use rame::layout::LayoutModel;
use rame::models::pp_doclayout::plus::{self, PpDocLayoutPlus};
use rame::sources::HuggingFace;

use crate::engine::PyOrtSessionConfig;
use crate::error::into_py_err;
use crate::image::array_to_image;
use crate::layout::PyLayoutResult;

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
        let artifact = plus::onnx::Artifact::default().session_config(session_config);
        let model = py
            .detach(|| {
                PpDocLayoutPlus::builder()
                    .source(hf_source)
                    .artifact(artifact)
                    .build()
            })
            .map_err(into_py_err)?;
        Ok(Self {
            inner: Box::new(model),
        })
    }

    fn detect_layout(
        &mut self,
        py: Python<'_>,
        image: PyReadonlyArray3<'_, u8>,
    ) -> PyResult<PyLayoutResult> {
        let img = array_to_image(image)?;
        py.detach(|| self.inner.detect_layout(&img))
            .map(PyLayoutResult::from)
            .map_err(into_py_err)
    }

    fn detect_layout_many(
        &mut self,
        py: Python<'_>,
        images: Vec<PyReadonlyArray3<'_, u8>>,
    ) -> PyResult<Vec<PyLayoutResult>> {
        let imgs: Vec<_> = images
            .into_iter()
            .map(array_to_image)
            .collect::<PyResult<_>>()?;
        py.detach(|| self.inner.detect_layout_many(&imgs))
            .map(|rs| rs.into_iter().map(PyLayoutResult::from).collect())
            .map_err(into_py_err)
    }
}
