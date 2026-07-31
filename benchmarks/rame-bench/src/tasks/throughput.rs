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
    warmup: usize,
    repeats: usize,
}

impl LayoutThroughputTask {
    pub fn new(
        dataset_root: impl Into<PathBuf>,
        batch_size: usize,
        warmup: usize,
        repeats: usize,
    ) -> BenchResult<Self> {
        if batch_size == 0 {
            return Err(BenchError::InvalidBatchSize);
        }
        if repeats == 0 {
            return Err(BenchError::InvalidRepeats);
        }

        Ok(Self {
            dataset: ImageDataset::new(dataset_root),
            batch_size,
            warmup,
            repeats,
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
        let images = samples
            .iter()
            .map(|sample| sample.load_image())
            .collect::<BenchResult<Vec<_>>>()?;

        for _ in 0..self.warmup {
            for batch in images.chunks(self.batch_size) {
                model.predict_many(batch)?;
            }
        }

        let started = Instant::now();
        let mut batches = 0u64;

        for _ in 0..self.repeats {
            for batch in images.chunks(self.batch_size) {
                model.predict_many(batch)?;
                batches += 1;
            }
        }

        let elapsed_s = started.elapsed().as_secs_f64();
        let total_samples = samples.len() * self.repeats;

        Ok(TaskReport::new(
            self.name(),
            vec![
                TaskMetric::integer("samples", samples.len() as u64),
                TaskMetric::integer("warmup", self.warmup as u64),
                TaskMetric::integer("repeats", self.repeats as u64),
                TaskMetric::integer("total_samples", total_samples as u64),
                TaskMetric::integer("batches", batches),
                TaskMetric::float("elapsed", elapsed_s, Some("s")),
                TaskMetric::float(
                    "throughput",
                    total_samples as f64 / elapsed_s.max(f64::MIN_POSITIVE),
                    Some("samples/s"),
                ),
            ],
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::error::BenchResult;
    use crate::models::LayoutModel;
    use crate::tasks::{BenchmarkTask, LayoutThroughputTask};
    use rame::image::Image;

    #[test]
    fn reports_throughput_metrics() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        write_test_image(root.path().join("a.png"))?;
        write_test_image(root.path().join("b.png"))?;
        write_test_image(root.path().join("c.png"))?;

        let mut model = CountingLayoutModel::default();
        let task = LayoutThroughputTask::new(root.path(), 2, 1, 2)?;

        let report = task.evaluate(&mut model)?;
        let metric_names = report
            .metrics()
            .iter()
            .map(|metric| metric.name())
            .collect::<Vec<_>>();

        assert_eq!(report.task(), crate::tasks::Task::LayoutThroughput);
        assert_eq!(
            metric_names,
            [
                "samples",
                "warmup",
                "repeats",
                "total_samples",
                "batches",
                "elapsed",
                "throughput"
            ]
        );
        assert_eq!(model.batches, 6);
        assert_eq!(model.images, 9);

        Ok(())
    }

    #[derive(Default)]
    struct CountingLayoutModel {
        batches: usize,
        images: usize,
    }

    impl LayoutModel for CountingLayoutModel {
        fn predict_many(&mut self, images: &[Image]) -> BenchResult<()> {
            self.batches += 1;
            self.images += images.len();
            Ok(())
        }
    }

    fn write_test_image(path: impl AsRef<std::path::Path>) -> Result<(), image::ImageError> {
        image::RgbImage::from_pixel(1, 1, image::Rgb([0, 0, 0])).save(path)
    }
}
