use clap::ValueEnum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum Model {
    #[value(name = "rame-pp-doclayout-plus-onnx")]
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
