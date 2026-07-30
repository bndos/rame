use std::fmt;
use std::str::FromStr;

use crate::datasets::ImageSample;
use crate::error::BenchResult;

use super::PpDocLayoutPlusOnnx;

const RAME_PP_DOCLAYOUT_PLUS_ONNX: &str = "rame-pp-doclayout-plus-onnx";

pub trait LayoutModel {
    fn predict_many(&mut self, samples: &[ImageSample]) -> BenchResult<()>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelName {
    RamePpDocLayoutPlusOnnx,
}

impl ModelName {
    pub const ALL: &'static [Self] = &[Self::RamePpDocLayoutPlusOnnx];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::RamePpDocLayoutPlusOnnx => RAME_PP_DOCLAYOUT_PLUS_ONNX,
        }
    }

    pub fn load_layout(self) -> BenchResult<Box<dyn LayoutModel>> {
        match self {
            Self::RamePpDocLayoutPlusOnnx => Ok(Box::new(PpDocLayoutPlusOnnx::new())),
        }
    }
}

impl fmt::Display for ModelName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseModelNameError(String);

impl fmt::Display for ParseModelNameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown model `{}`", self.0)
    }
}

impl std::error::Error for ParseModelNameError {}

impl FromStr for ModelName {
    type Err = ParseModelNameError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            RAME_PP_DOCLAYOUT_PLUS_ONNX => Ok(Self::RamePpDocLayoutPlusOnnx),
            _ => Err(ParseModelNameError(value.to_string())),
        }
    }
}
