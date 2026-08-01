mod bench_metrics;
mod cli;
pub mod datasets;
pub mod error;
pub mod models;
pub mod tasks;

fn main() -> error::BenchResult<()> {
    cli::run()?;
    Ok(())
}
