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
    PageNumber,
    Abstract,
    Content,
    Figure,
    FigureCaption,
    Image,
    Chart,
    Table,
    TableCaption,
    Header,
    HeaderImage,
    Footer,
    FooterImage,
    Footnote,
    Formula,
    FormulaNumber,
    Algorithm,
    Reference,
    ReferenceContent,
    AsideText,
    Seal,
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
