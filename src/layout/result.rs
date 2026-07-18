use crate::layout::Geometry;

/// Result of a document layout detection run.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutResult {
    pub regions: Vec<LayoutRegion>,
}

impl LayoutResult {
    pub fn new(regions: Vec<LayoutRegion>) -> Self {
        Self { regions }
    }
}

/// Semantic label assigned to a layout region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutLabel {
    Text,
    Title,
    Figure,
    FigureCaption,
    Table,
    TableCaption,
    Header,
    Footer,
    Formula,
    Reference,
    Unknown(String),
}

/// A detected document layout region.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutRegion {
    pub label: LayoutLabel,
    pub score: f32,
    pub geometry: Geometry,
    /// Optional zero-based reading order emitted by models that provide it.
    pub reading_order: Option<usize>,
}
