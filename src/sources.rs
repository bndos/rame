pub use rame_core::sources::{ResolveModelSource, ResolvedModelSource, SourceError};

#[cfg(feature = "huggingface")]
mod huggingface;

#[cfg(feature = "huggingface")]
pub use huggingface::{HuggingFace, HuggingFaceModel};
