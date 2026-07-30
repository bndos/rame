use clap::ValueEnum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum Task {
    #[value(name = "layout-throughput")]
    LayoutThroughput,
}

impl Task {
    pub const ALL: &'static [Self] = &[Self::LayoutThroughput];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::LayoutThroughput => "layout-throughput",
        }
    }
}
