use std::path::Path;

use rame::image::Image;
use rame::layout::{Geometry, LayoutModel, LayoutRegion};
use rame::models::pp_doclayout::plus;
use rame::runtime::ModelLoader;
use rame::sources::HuggingFace;
use tabled::settings::Style;
use tabled::{Table, Tabled};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = load_rgb_image("examples/fixtures/document.png")?;
    let source = HuggingFace::new()?.model("PaddlePaddle/PP-DocLayout_plus-L_onnx");

    let mut model = plus::onnx::Loader::default().load(source)?;

    let result = model.detect_layout(&image)?;

    print_regions(&result.regions);

    Ok(())
}

fn load_rgb_image(path: impl AsRef<Path>) -> Result<Image, Box<dyn std::error::Error>> {
    let image = image::ImageReader::open(path)?.decode()?.to_rgb8();
    let (width, height) = image.dimensions();

    Ok(Image::from_rgb8(width, height, image.into_raw())?)
}

fn print_regions(regions: &[LayoutRegion]) {
    let rows = regions
        .iter()
        .take(10)
        .enumerate()
        .map(DetectionRow::from_region)
        .collect::<Vec<_>>();

    println!("PP-DocLayout Plus ONNX detections");
    println!("{}", Table::new(rows).with(Style::markdown()));
}

#[derive(Tabled)]
struct DetectionRow {
    #[tabled(rename = "#")]
    index: usize,
    label: String,
    score: String,
    x_min: String,
    y_min: String,
    x_max: String,
    y_max: String,
}

impl DetectionRow {
    fn from_region((index, region): (usize, &LayoutRegion)) -> Self {
        match &region.geometry {
            Geometry::Rect(rect) => Self {
                index: index + 1,
                label: format!("{:?}", region.label),
                score: format!("{:.3}", region.score),
                x_min: format!("{:.1}", rect.x_min),
                y_min: format!("{:.1}", rect.y_min),
                x_max: format!("{:.1}", rect.x_max),
                y_max: format!("{:.1}", rect.y_max),
            },
            Geometry::Polygon(polygon) => Self {
                index: index + 1,
                label: format!("{:?}", region.label),
                score: format!("{:.3}", region.score),
                x_min: "polygon".to_string(),
                y_min: polygon.points.len().to_string(),
                x_max: String::new(),
                y_max: String::new(),
            },
        }
    }
}
