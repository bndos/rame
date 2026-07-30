#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Model {
    RamePpDocLayoutPlusOnnx,
}

impl Model {
    pub const ALL: &'static [Self] = &[Self::RamePpDocLayoutPlusOnnx];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::RamePpDocLayoutPlusOnnx => "rame-pp-doclayout-plus-onnx",
        }
    }
}
