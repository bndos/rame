mod cli;
pub mod datasets;
pub mod error;
pub mod models;

fn main() -> error::BenchResult<()> {
    cli::run()?;
    Ok(())
}
