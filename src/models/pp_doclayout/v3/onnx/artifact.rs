use std::path::PathBuf;

use crate::models::pp_doclayout::v3::decoder::PpDocLayoutV3Decoder;
use crate::models::pp_doclayout::v3::model::PpDocLayoutV3;
use crate::models::pp_doclayout::v3::onnx::processor::PpDocLayoutV3OnnxProcessor;
use crate::preprocess::PreprocessConfig;
use crate::preprocess::vision::{Interpolation, NormalizeImage, Resize, ToTensor};
use crate::runtime::{ArtifactParts, ModelArtifact};
use crate::session::ort::OrtBackend;
use crate::session::ort::OrtSessionConfig;

/// PaddleOCR PP-DocLayoutV3 ONNX artifact configuration.
///
/// The official ONNX artifact uses `inference.onnx`, Paddle-style inputs
/// `image`, `im_shape`, `scale_factor`, and fixed 800x800 preprocessing with
/// PaddleX object-detection scale-only normalization.
///
/// Sources:
/// - PP-DocLayoutV3 transform config: <https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/repo_apis/PaddleDetection_api/configs/PP-DocLayoutV3.yaml>
/// - PaddleX `NormalizeImage` defaults `is_scale` to `true`, making `norm_type: none` scale by `1 / 255`: <https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/inference/models/object_detection/predictor.py>
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
    pub masks: String,
}

impl Default for Outputs {
    fn default() -> Self {
        Self {
            boxes: "fetch_name_0".to_string(),
            boxes_num: "fetch_name_1".to_string(),
            masks: "fetch_name_2".to_string(),
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
    type Architecture = PpDocLayoutV3;
    type Backend = OrtBackend;
    type Processor = PpDocLayoutV3OnnxProcessor;
    type Decoder = PpDocLayoutV3Decoder;

    fn into_parts(
        self,
    ) -> ArtifactParts<OrtSessionConfig, PpDocLayoutV3OnnxProcessor, PpDocLayoutV3Decoder> {
        let session_config = self
            .session_config
            .output(self.outputs.boxes.clone())
            .output(self.outputs.boxes_num.clone());

        ArtifactParts {
            model_file: self.model_file,
            session_config,
            processor: PpDocLayoutV3OnnxProcessor::new(
                self.inputs,
                self.preprocess,
                self.preprocess_config,
            ),
            decoder: PpDocLayoutV3Decoder::new(self.outputs.boxes, self.outputs.boxes_num),
        }
    }
}
