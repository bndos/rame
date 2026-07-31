use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::datasets::DatasetError;
use crate::error::BenchResult;

const IMAGE_EXTENSIONS: &[&str] = &["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSample {
    path: PathBuf,
}

impl ImageSample {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load_image(&self) -> BenchResult<rame::image::Image> {
        let image = image::ImageReader::open(self.path())?
            .with_guessed_format()?
            .decode()?
            .to_rgb8();
        let (width, height) = image.dimensions();

        Ok(rame::image::Image::from_rgb8(
            width,
            height,
            image.into_raw(),
        )?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageDataset {
    root: PathBuf,
}

impl ImageDataset {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn samples(&self) -> Result<Vec<ImageSample>, DatasetError> {
        let mut paths = image_paths(&self.root)?;
        paths.sort();

        Ok(paths.into_iter().map(ImageSample::new).collect())
    }
}

fn image_paths(root: &Path) -> Result<Vec<PathBuf>, DatasetError> {
    if root.is_file() {
        if is_image(root) {
            return Ok(vec![root.to_path_buf()]);
        }

        return Ok(Vec::new());
    }

    let mut images = Vec::new();
    for entry in WalkDir::new(root) {
        let entry = entry.map_err(|source| DatasetError::Walk {
            root: root.to_path_buf(),
            source,
        })?;

        let path = entry.path();
        if entry.file_type().is_file() && is_image(path) {
            images.push(path.to_path_buf());
        }
    }

    Ok(images)
}

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            IMAGE_EXTENSIONS
                .iter()
                .any(|candidate| extension.eq_ignore_ascii_case(candidate))
        })
}
