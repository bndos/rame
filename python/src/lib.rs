use std::sync::OnceLock;

use pyo3::prelude::*;

mod engine;
mod error;
mod image;
mod layout;
mod models;

pub fn get_rame_python_version() -> &'static str {
    static RAME_PYTHON_VERSION: OnceLock<String> = OnceLock::new();

    RAME_PYTHON_VERSION.get_or_init(|| {
        let version = env!("CARGO_PKG_VERSION");
        // Match pydantic-core's lightweight Rust semver to Python prerelease
        // spelling conversion: https://github.com/pydantic/pydantic/blob/main/pydantic-core/src/lib.rs
        version.replace("-alpha", "a").replace("-beta", "b")
    })
}

pub fn build_info() -> String {
    format!(
        "profile={} pgo={}",
        env!("PROFILE"),
        cfg!(specified_profile_use)
    )
}

#[pymodule]
mod _native {
    #[allow(clippy::wildcard_imports)]
    use super::*;

    #[pymodule_export]
    use crate::engine::{PyOrtCpuConfig, PyOrtCudaConfig, PyOrtSessionConfig, PyOrtTrtConfig};
    #[pymodule_export]
    use crate::layout::{PyGeometry, PyLayoutRegion, PyLayoutResult};
    #[pymodule_export]
    use crate::models::pp_doclayout::{PyPpDocLayoutPlusOnnx, PyPpDocLayoutV3Onnx};

    #[pymodule_init]
    fn module_init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("__version__", get_rame_python_version())?;
        module.add("build_info", build_info())?;
        Ok(())
    }
}
