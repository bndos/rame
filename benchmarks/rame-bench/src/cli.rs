use clap::{Parser, Subcommand};

use crate::error::BenchResult;
use crate::models::Model;
use crate::tasks::Task;

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
    Tasks,
}

pub fn run() -> BenchResult<()> {
    match Cli::parse().command {
        Command::Models => list_models(),
        Command::Tasks => list_tasks(),
    }

    Ok(())
}

fn list_models() {
    for model in Model::ALL {
        println!("{}", model.as_str());
    }
}

fn list_tasks() {
    for task in Task::ALL {
        println!("{}", task.as_str());
    }
}
