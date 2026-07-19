mod error;
mod model_source;

#[cfg(feature = "huggingface")]
mod huggingface;

pub use error::SourceError;
#[cfg(feature = "huggingface")]
pub use huggingface::{HuggingFace, HuggingFaceModel};
pub use model_source::ResolveModelSource;
