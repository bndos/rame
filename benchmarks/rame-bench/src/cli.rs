use clap::{Parser, Subcommand};

use crate::models::Model;

#[derive(Debug, Parser)]
#[command(name = "rame-bench")]
#[command(about = "Benchmark runner for rame models")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Models,
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Models => list_models(),
    }

    Ok(())
}

fn list_models() {
    for model in Model::ALL {
        println!("{}", model.as_str());
    }
}
