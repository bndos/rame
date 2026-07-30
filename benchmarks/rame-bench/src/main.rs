mod cli;
pub mod datasets;
pub mod models;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cli::run()?;
    Ok(())
}
