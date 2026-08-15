use std::path::PathBuf;

use crate::models::pp_doclayout::plus::decoder::PpDocLayoutPlusDecoder;
use crate::models::pp_doclayout::plus::model::PpDocLayoutPlus;
use crate::models::pp_doclayout::plus::onnx::processor::PpDocLayoutPlusOnnxProcessor;
use crate::preprocess::PreprocessConfig;
use crate::preprocess::vision::{Interpolation, NormalizeImage, Resize, ToTensor};
use crate::runtime::{ArtifactParts, ModelArtifact};
use crate::session::ort::OrtBackend;
use crate::session::ort::OrtSessionConfig;

/// PaddleX PP-DocLayout Plus ONNX artifact configuration.
///
/// The preprocessing recipe follows PaddleX RT-DETR metadata for
/// `PP-DocLayout_plus-L`: resize to 800x800 and convert to NCHW f32 tensor
/// with scale-only normalization.
/// Source: <https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/modules/base/utils/pdparams2safetensors/inference_meta.py#L150-L159>
#[derive(Debug, Clone)]
pub struct Artifact {
    pub model_file: PathBuf,
    pub inputs: Inputs,
    pub outputs: Outputs,
    pub preprocess: Preprocess,
    pub preprocess_config: PreprocessConfig,
    pub session_config: OrtSessionConfig,
}

impl Default for Artifact {
    fn default() -> Self {
        Self {
            model_file: PathBuf::from("inference.onnx"),
            inputs: Inputs::default(),
            outputs: Outputs::default(),
            preprocess: Preprocess::default(),
            preprocess_config: PreprocessConfig::default(),
            session_config: OrtSessionConfig::default(),
        }
    }
}

impl Artifact {
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

    pub fn preprocess_config(mut self, config: PreprocessConfig) -> Self {
        self.preprocess_config = config;
        self
    }

    pub fn session_config(mut self, session_config: OrtSessionConfig) -> Self {
        self.session_config = session_config;
        self
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

impl ModelArtifact for Artifact {
    type Architecture = PpDocLayoutPlus;
    type Backend = OrtBackend;
    type Processor = PpDocLayoutPlusOnnxProcessor;
    type Decoder = PpDocLayoutPlusDecoder;

    fn into_parts(
        self,
    ) -> ArtifactParts<OrtSessionConfig, PpDocLayoutPlusOnnxProcessor, PpDocLayoutPlusDecoder> {
        let session_config = self
            .session_config
            .output(self.outputs.boxes.clone())
            .output(self.outputs.boxes_num.clone());

        ArtifactParts {
            model_file: self.model_file,
            session_config,
            processor: PpDocLayoutPlusOnnxProcessor::new(
                self.inputs,
                self.preprocess,
                self.preprocess_config,
            ),
            decoder: PpDocLayoutPlusDecoder::new(self.outputs.boxes, self.outputs.boxes_num),
        }
    }
}
