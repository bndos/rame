use std::path::PathBuf;
use std::time::Instant;

use crate::datasets::ImageDataset;
use crate::error::{BenchError, BenchResult};
use crate::models::LayoutModel;
use crate::tasks::{BenchmarkTask, Task, TaskMetric, TaskReport};

#[derive(Debug, Clone)]
pub struct LayoutThroughputTask {
    dataset: ImageDataset,
    batch_size: usize,
}

impl LayoutThroughputTask {
    pub fn new(dataset_root: impl Into<PathBuf>, batch_size: usize) -> BenchResult<Self> {
        if batch_size == 0 {
            return Err(BenchError::InvalidBatchSize);
        }

        Ok(Self {
            dataset: ImageDataset::new(dataset_root),
            batch_size,
        })
    }
}

impl BenchmarkTask for LayoutThroughputTask {
    type Model = dyn LayoutModel;

    fn name(&self) -> Task {
        Task::LayoutThroughput
    }

    fn evaluate(&self, model: &mut Self::Model) -> BenchResult<TaskReport> {
        let samples = self.dataset.samples()?;
        if samples.is_empty() {
            return Err(BenchError::EmptyDataset);
        }

        let started = Instant::now();
        let mut batches = 0u64;

        for batch in samples.chunks(self.batch_size) {
            model.predict_many(batch)?;
            batches += 1;
        }

        let elapsed_s = started.elapsed().as_secs_f64();

        Ok(TaskReport::new(
            self.name(),
            vec![
                TaskMetric::integer("samples", samples.len() as u64),
                TaskMetric::integer("batches", batches),
                TaskMetric::float("elapsed", elapsed_s, Some("s")),
                TaskMetric::float(
                    "throughput",
                    samples.len() as f64 / elapsed_s.max(f64::MIN_POSITIVE),
                    Some("samples/s"),
                ),
            ],
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::models::PpDocLayoutPlusOnnx;
    use crate::tasks::{BenchmarkTask, LayoutThroughputTask};

    #[test]
    fn reports_throughput_metrics() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        fs::write(root.path().join("a.png"), [])?;
        fs::write(root.path().join("b.png"), [])?;
        fs::write(root.path().join("c.png"), [])?;

        let mut model = PpDocLayoutPlusOnnx::new();
        let task = LayoutThroughputTask::new(root.path(), 2)?;

        let report = task.evaluate(&mut model)?;
        let metric_names = report
            .metrics()
            .iter()
            .map(|metric| metric.name())
            .collect::<Vec<_>>();

        assert_eq!(report.task(), crate::tasks::Task::LayoutThroughput);
        assert_eq!(
            metric_names,
            ["samples", "batches", "elapsed", "throughput"]
        );

        Ok(())
    }
}
