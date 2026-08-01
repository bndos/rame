#[cfg(feature = "profile")]
mod imp {
    use crate::error::{BenchError, BenchResult};

    pub fn install() -> BenchResult<()> {
        tracing_subscriber::fmt()
            .json()
            .with_writer(std::io::stderr)
            .with_ansi(false)
            .init();

        metrics::set_global_recorder(
            metrics_exporter_tracing::TracingRecorder::builder()
                .default_target("rame_bench_metrics")
                .build(),
        )
        .map_err(|_| BenchError::MetricsRecorderAlreadyInstalled)?;

        Ok(())
    }
}

#[cfg(not(feature = "profile"))]
mod imp {
    use crate::error::{BenchError, BenchResult};

    pub fn install() -> BenchResult<()> {
        Err(BenchError::ProfileFeatureDisabled)
    }
}

pub use imp::install;
