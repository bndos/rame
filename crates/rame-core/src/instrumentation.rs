#[cfg(feature = "metrics")]
macro_rules! time_stage {
    ($name:literal, $expr:expr) => {{
        let started = std::time::Instant::now();
        let result = $expr;
        metrics::histogram!($name).record(started.elapsed().as_secs_f64());
        result
    }};
}

#[cfg(not(feature = "metrics"))]
macro_rules! time_stage {
    ($name:literal, $expr:expr) => {{
        let _ = $name;
        $expr
    }};
}

pub(crate) use time_stage;
