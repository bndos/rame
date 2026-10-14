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

pub mod backends;
pub mod models;
pub mod preprocess;
pub mod registry;
pub mod runtime;
pub mod session;
pub mod sources;
pub mod tokenization;
pub use rame_core::{
    Device, RameError, RameResult, audio, error, geometry, image, layout, ocr, tensor,
    transcription,
};
