use ort::session::builder::SessionBuilder;

use crate::session::ort::OrtError;

/// ONNX Runtime session configuration.
#[derive(Debug, Clone, Default)]
pub struct OrtSessionConfig {
    pub(super) intra_threads: Option<usize>,
    pub(super) inter_threads: Option<usize>,
    pub(super) execution_provider: OrtExecutionProvider,
}

impl OrtSessionConfig {
    pub fn intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = Some(threads);
        self
    }

    pub fn inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = Some(threads);
        self
    }

    #[cfg(feature = "onnxruntime-cuda")]
    pub fn cuda(mut self, device_id: i32) -> Self {
        self.execution_provider = OrtExecutionProvider::Cuda { device_id };
        self
    }

    #[cfg(feature = "onnxruntime-tensorrt")]
    pub fn tensorrt(mut self, device_id: i32) -> Self {
        self.execution_provider = OrtExecutionProvider::TensorRt {
            device_id,
            fp16: false,
        };
        self
    }

    #[cfg(feature = "onnxruntime-tensorrt")]
    pub fn tensorrt_fp16(mut self, device_id: i32) -> Self {
        self.execution_provider = OrtExecutionProvider::TensorRt {
            device_id,
            fp16: true,
        };
        self
    }

    pub(super) fn apply(self, mut builder: SessionBuilder) -> Result<SessionBuilder, OrtError> {
        builder = match self.execution_provider {
            OrtExecutionProvider::Cpu => builder,
            #[cfg(feature = "onnxruntime-cuda")]
            OrtExecutionProvider::Cuda { device_id } => builder
                .with_execution_providers([ort::ep::CUDA::default()
                    .with_device_id(device_id)
                    .build()
                    .error_on_failure()])
                .map_err(OrtError::from)?,
            #[cfg(feature = "onnxruntime-tensorrt")]
            OrtExecutionProvider::TensorRt { device_id, fp16 } => builder
                .with_execution_providers([
                    ort::ep::TensorRT::default()
                        .with_device_id(device_id)
                        .with_fp16(fp16)
                        .build()
                        .error_on_failure(),
                    ort::ep::CUDA::default()
                        .with_device_id(device_id)
                        .build()
                        .error_on_failure(),
                ])
                .map_err(OrtError::from)?,
        };

        if let Some(threads) = self.intra_threads {
            builder = builder
                .with_intra_threads(threads)
                .map_err(OrtError::from)?;
        }

        if let Some(threads) = self.inter_threads {
            builder = builder
                .with_inter_threads(threads)
                .map_err(OrtError::from)?;
        }

        Ok(builder)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) enum OrtExecutionProvider {
    #[default]
    Cpu,
    #[cfg(feature = "onnxruntime-cuda")]
    Cuda { device_id: i32 },
    #[cfg(feature = "onnxruntime-tensorrt")]
    TensorRt { device_id: i32, fp16: bool },
}
