use std::fmt;
use std::str::FromStr;

use crate::datasets::ImageSample;
use crate::error::BenchResult;

const RAME_PP_DOCLAYOUT_PLUS_ONNX: &str = "rame-pp-doclayout-plus-onnx";

pub trait LayoutModel {
    fn predict_many(&mut self, samples: &[ImageSample]) -> BenchResult<()>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Model {
    RamePpDocLayoutPlusOnnx,
}

impl Model {
    pub const ALL: &'static [Self] = &[Self::RamePpDocLayoutPlusOnnx];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::RamePpDocLayoutPlusOnnx => RAME_PP_DOCLAYOUT_PLUS_ONNX,
        }
    }
}

impl fmt::Display for Model {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseModelError(String);

impl fmt::Display for ParseModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown model `{}`", self.0)
    }
}

impl std::error::Error for ParseModelError {}

impl FromStr for Model {
    type Err = ParseModelError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            RAME_PP_DOCLAYOUT_PLUS_ONNX => Ok(Self::RamePpDocLayoutPlusOnnx),
            _ => Err(ParseModelError(value.to_string())),
        }
    }
}
