use crate::geometry::{Polygon, Rect};

/// Geometry describing a detected layout region.
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    Rect(Rect),
    Polygon(Polygon),
}
