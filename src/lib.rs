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

pub mod error;
pub mod image;
mod instrumentation;
pub mod layout;
pub mod models;
pub mod preprocess;
pub mod runtime;
pub mod session;
pub mod sources;
pub mod tensor;

pub use error::{RameError, RameResult};
pub use tensor::Device;
