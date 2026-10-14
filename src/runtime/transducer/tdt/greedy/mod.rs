mod standard;
mod workspace;

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use crate::backends::mlx::tdt::MlxTdtGreedyExecutor;
pub use standard::StandardTdtGreedyExecutor;
