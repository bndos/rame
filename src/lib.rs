#![cfg_attr(
    not(test),
    deny(
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::unwrap_used
    )
)]

pub mod audio;
pub mod error;
pub mod geometry;
pub mod image;
mod instrumentation;
pub mod layout;
pub mod models;
pub mod ocr;
pub mod preprocess;
pub mod runtime;
pub mod session;
pub mod sources;
pub mod tensor;
pub mod tokenization;
pub mod transcription;

pub use error::{RameError, RameResult};
pub use tensor::Device;
