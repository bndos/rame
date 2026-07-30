use std::fmt;
use std::str::FromStr;

use serde::Serialize;

const LAYOUT_THROUGHPUT: &str = "layout-throughput";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Task {
    LayoutThroughput,
}

impl Task {
    pub const ALL: &'static [Self] = &[Self::LayoutThroughput];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::LayoutThroughput => LAYOUT_THROUGHPUT,
        }
    }
}

impl fmt::Display for Task {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for Task {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseTaskError(String);

impl fmt::Display for ParseTaskError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown task `{}`", self.0)
    }
}

impl std::error::Error for ParseTaskError {}

impl FromStr for Task {
    type Err = ParseTaskError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            LAYOUT_THROUGHPUT => Ok(Self::LayoutThroughput),
            _ => Err(ParseTaskError(value.to_string())),
        }
    }
}
