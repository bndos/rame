use std::path::PathBuf;

use crate::RameResult;
use crate::image::ImageView;
use crate::models::pp_doclayout::plus::decoder::PpDocLayoutPlusDecoder;
use crate::models::pp_doclayout::plus::onnx::processor::PpDocLayoutPlusOnnxProcessor;
use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp, PreprocessPipelineBuilder};
use crate::preprocess::vision::{
    Interpolation, NormalizeImage, Resize, ToTensor, VisionBatchOutput,
};
use crate::runtime::{ModelLoader, StandardModelRunner};
use crate::session::SessionBackend;
use crate::session::ort::{OrtBackend, OrtSession, OrtSessionConfig};
use crate::sources::ResolvedModelSource;

/// Loads the PaddleX PP-DocLayout Plus ONNX export.
///
/// The preprocessing recipe follows PaddleX RT-DETR metadata for
/// `PP-DocLayout_plus-L`: resize to 800x800 and convert to NCHW f32 tensor
/// with scale-only normalization.
/// Source: <https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/modules/base/utils/pdparams2safetensors/inference_meta.py#L150-L159>
#[derive(Debug, Clone)]
pub struct Loader<B> {
    pub model_file: PathBuf,
    pub inputs: Inputs,
    pub outputs: Outputs,
    pub preprocess: Preprocess,
    pub preprocess_backend: B,
    pub session_config: OrtSessionConfig,
}

impl<B> Loader<B> {
    pub fn with_backend(preprocess_backend: B) -> Self {
        Self {
            model_file: PathBuf::from("inference.onnx"),
            inputs: Inputs::default(),
            outputs: Outputs::default(),
            preprocess: Preprocess::default(),
            preprocess_backend,
            session_config: OrtSessionConfig::default(),
        }
    }

    pub fn preprocess_backend<C>(self, preprocess_backend: C) -> Loader<C> {
        Loader {
            model_file: self.model_file,
            inputs: self.inputs,
            outputs: self.outputs,
            preprocess: self.preprocess,
            preprocess_backend,
            session_config: self.session_config,
        }
    }

    pub fn model_file(mut self, model_file: impl Into<PathBuf>) -> Self {
        self.model_file = model_file.into();
        self
    }

    pub fn inputs(mut self, inputs: Inputs) -> Self {
        self.inputs = inputs;
        self
    }

    pub fn outputs(mut self, outputs: Outputs) -> Self {
        self.outputs = outputs;
        self
    }

    pub fn preprocess(mut self, preprocess: Preprocess) -> Self {
        self.preprocess = preprocess;
        self
    }

    pub fn session_config(mut self, session_config: OrtSessionConfig) -> Self {
        self.session_config = session_config;
        self
    }
}

#[cfg(feature = "opencv")]
impl Default for Loader<crate::backends::opencv::OpenCvVisionBackend> {
    fn default() -> Self {
        Self::with_backend(crate::backends::opencv::OpenCvVisionBackend::default())
    }
}

#[cfg(feature = "opencv")]
impl Loader<crate::backends::opencv::OpenCvVisionBackend> {
    pub fn preprocess_config(self, config: crate::preprocess::PreprocessConfig) -> Self {
        self.preprocess_backend(config.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inputs {
    pub image: String,
    pub im_shape: String,
    pub scale_factor: String,
}

impl Default for Inputs {
    fn default() -> Self {
        Self {
            image: "image".to_string(),
            im_shape: "im_shape".to_string(),
            scale_factor: "scale_factor".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outputs {
    pub boxes: String,
    pub boxes_num: String,
}

impl Default for Outputs {
    fn default() -> Self {
        Self {
            boxes: "fetch_name_0".to_string(),
            boxes_num: "fetch_name_1".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Preprocess {
    pub resize: Resize,
    pub tensor: ToTensor,
}

impl Default for Preprocess {
    fn default() -> Self {
        Self {
            resize: Resize::fixed_square(800, Interpolation::Cubic),
            tensor: ToTensor::nchw().normalize(NormalizeImage::scale(NormalizeImage::INV_255)),
        }
    }
}

impl<B> ModelLoader for Loader<B>
where
    B: for<'a> PreprocessBackend<Source<'a> = ImageView<'a>, Output = VisionBatchOutput>,
    Resize: PreprocessOp<B>,
    ToTensor: PreprocessOp<B>,
{
    type Runner =
        StandardModelRunner<PpDocLayoutPlusOnnxProcessor<B>, OrtSession, PpDocLayoutPlusDecoder>;

    fn load_resolved(self, source: ResolvedModelSource) -> RameResult<Self::Runner> {
        let session_config = self
            .session_config
            .output(self.outputs.boxes.clone())
            .output(self.outputs.boxes_num.clone());
        let model_path = source.join_artifact_path(&self.model_file)?;
        let session = OrtBackend::load(&model_path, session_config)?;

        let pipeline = PreprocessPipelineBuilder::new(self.preprocess_backend)
            .add_op(self.preprocess.resize)
            .add_op(self.preprocess.tensor)
            .compile();

        Ok(StandardModelRunner::new(
            PpDocLayoutPlusOnnxProcessor::with_pipeline(self.inputs, pipeline),
            session,
            PpDocLayoutPlusDecoder::new(self.outputs.boxes, self.outputs.boxes_num),
        ))
    }
}

#[cfg(all(test, not(feature = "opencv")))]
mod tests {
    use crate::RameResult;
    use crate::image::ImageView;
    use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp};
    use crate::preprocess::vision::{Resize, ToTensor, VisionBatchOutput};
    use crate::runtime::ModelLoader;

    use super::Loader;

    #[derive(Debug)]
    struct OtherVisionBackend;

    impl PreprocessBackend for OtherVisionBackend {
        type Source<'a> = ImageView<'a>;
        type Data<'a> = ();
        type Output = VisionBatchOutput;

        fn input<'a>(&self, _: &'a [ImageView<'a>]) -> RameResult<Self::Data<'a>> {
            Ok(())
        }

        fn finish(&self, _: Self::Data<'_>) -> RameResult<Self::Output> {
            Err(crate::preprocess::PreprocessError::MissingOutput.into())
        }
    }

    impl PreprocessOp<OtherVisionBackend> for Resize {
        fn forward<'a>(
            &self,
            data: <OtherVisionBackend as PreprocessBackend>::Data<'a>,
        ) -> RameResult<<OtherVisionBackend as PreprocessBackend>::Data<'a>> {
            Ok(data)
        }
    }

    impl PreprocessOp<OtherVisionBackend> for ToTensor {
        fn forward<'a>(
            &self,
            data: <OtherVisionBackend as PreprocessBackend>::Data<'a>,
        ) -> RameResult<<OtherVisionBackend as PreprocessBackend>::Data<'a>> {
            Ok(data)
        }
    }

    #[test]
    fn onnx_loader_accepts_an_explicit_non_opencv_preprocessor() {
        fn assert_model_loader<L: ModelLoader>(_: &L) {}

        let loader = Loader::with_backend(OtherVisionBackend);
        assert_model_loader(&loader);
    }
}
