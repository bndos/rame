#[cfg(all(feature = "mlx", target_os = "macos"))]
mod mlx;
mod standard;
mod workspace;

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use mlx::MlxTdtGreedyExecutor;
pub use standard::StandardTdtGreedyExecutor;
