use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::error::BenchResult;
use crate::models::ModelName;
use crate::tasks::{BenchmarkTask, LayoutThroughputTask, Task};

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
    Run {
        #[arg(long)]
        model: ModelName,
        #[arg(long)]
        task: Task,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value_t = 0)]
        warmup: usize,
        #[arg(long, default_value_t = 1)]
        repeats: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Tasks,
}

pub fn run() -> BenchResult<()> {
    match Cli::parse().command {
        Command::Models => list_models(),
        Command::Run {
            model,
            task,
            dataset,
            batch_size,
            warmup,
            repeats,
            output,
        } => run_task(model, task, dataset, batch_size, warmup, repeats, output)?,
        Command::Tasks => list_tasks(),
    }

    Ok(())
}

fn list_models() {
    for model in ModelName::ALL {
        println!("{model}");
    }
}

fn list_tasks() {
    for task in Task::ALL {
        println!("{task}");
    }
}

fn run_task(
    model: ModelName,
    task: Task,
    dataset: PathBuf,
    batch_size: usize,
    warmup: usize,
    repeats: usize,
    output: Option<PathBuf>,
) -> BenchResult<()> {
    let report = match task {
        Task::LayoutThroughput => {
            let mut model = model.load_layout()?;
            LayoutThroughputTask::new(dataset, batch_size, warmup, repeats)?
                .evaluate(model.as_mut())?
        }
    };

    report.write_json(std::io::stdout().lock())?;
    println!();
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        report.write_json(std::fs::File::create(path)?)?;
    }

    Ok(())
}
