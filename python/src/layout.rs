use pyo3::prelude::*;

use rame::layout::{Geometry, LayoutLabel, LayoutRegion, LayoutResult};

/// Result of a document layout detection run.
#[pyclass(name = "LayoutResult", get_all, skip_from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct PyLayoutResult {
    regions: Vec<PyLayoutRegion>,
}

impl From<LayoutResult> for PyLayoutResult {
    fn from(result: LayoutResult) -> Self {
        Self {
            regions: result
                .regions
                .into_iter()
                .map(PyLayoutRegion::from)
                .collect(),
        }
    }
}

/// A detected document layout region.
#[pyclass(name = "LayoutRegion", get_all, skip_from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct PyLayoutRegion {
    label: String,
    score: f32,
    geometry: PyGeometry,
    reading_order: Option<usize>,
}

impl From<LayoutRegion> for PyLayoutRegion {
    fn from(region: LayoutRegion) -> Self {
        Self {
            label: label_name(region.label),
            score: region.score,
            geometry: region.geometry.into(),
            reading_order: region.reading_order,
        }
    }
}

/// Geometry describing a detected layout region.
#[pyclass(name = "Geometry", get_all, skip_from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct PyGeometry {
    kind: String,
    coordinates: Vec<f32>,
}

impl From<Geometry> for PyGeometry {
    fn from(geometry: Geometry) -> Self {
        match geometry {
            Geometry::Rect(rect) => Self {
                kind: "rect".to_string(),
                coordinates: vec![rect.x_min, rect.y_min, rect.x_max, rect.y_max],
            },
            Geometry::Polygon(polygon) => Self {
                kind: "polygon".to_string(),
                coordinates: polygon
                    .points
                    .into_iter()
                    .flat_map(|point| [point.x, point.y])
                    .collect(),
            },
        }
    }
}

fn label_name(label: LayoutLabel) -> String {
    match label {
        LayoutLabel::Text => "text".to_string(),
        LayoutLabel::Title => "title".to_string(),
        LayoutLabel::PageNumber => "page_number".to_string(),
        LayoutLabel::Abstract => "abstract".to_string(),
        LayoutLabel::Content => "content".to_string(),
        LayoutLabel::Figure => "figure".to_string(),
        LayoutLabel::FigureCaption => "figure_caption".to_string(),
        LayoutLabel::Image => "image".to_string(),
        LayoutLabel::Chart => "chart".to_string(),
        LayoutLabel::Table => "table".to_string(),
        LayoutLabel::TableCaption => "table_caption".to_string(),
        LayoutLabel::Header => "header".to_string(),
        LayoutLabel::HeaderImage => "header_image".to_string(),
        LayoutLabel::Footer => "footer".to_string(),
        LayoutLabel::FooterImage => "footer_image".to_string(),
        LayoutLabel::Footnote => "footnote".to_string(),
        LayoutLabel::Formula => "formula".to_string(),
        LayoutLabel::FormulaNumber => "formula_number".to_string(),
        LayoutLabel::Algorithm => "algorithm".to_string(),
        LayoutLabel::Reference => "reference".to_string(),
        LayoutLabel::ReferenceContent => "reference_content".to_string(),
        LayoutLabel::AsideText => "aside_text".to_string(),
        LayoutLabel::Seal => "seal".to_string(),
        LayoutLabel::Unknown(label) => label,
    }
}
