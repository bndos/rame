use std::fmt;
use std::io::Write;

use serde::Serialize;

use crate::error::BenchResult;
use crate::tasks::Task;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TaskReport {
    task: Task,
    metrics: Vec<TaskMetric>,
}

impl TaskReport {
    pub fn new(task: Task, metrics: Vec<TaskMetric>) -> Self {
        Self { task, metrics }
    }

    pub fn task(&self) -> Task {
        self.task
    }

    pub fn metrics(&self) -> &[TaskMetric] {
        &self.metrics
    }

    pub fn write_json(&self, writer: impl Write) -> BenchResult<()> {
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TaskMetric {
    name: &'static str,
    value: MetricValue,
    unit: Option<&'static str>,
}

impl TaskMetric {
    pub fn integer(name: &'static str, value: u64) -> Self {
        Self {
            name,
            value: MetricValue::Integer(value),
            unit: None,
        }
    }

    pub fn float(name: &'static str, value: f64, unit: impl Into<Option<&'static str>>) -> Self {
        Self {
            name,
            value: MetricValue::Float(value),
            unit: unit.into(),
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn value(&self) -> MetricValue {
        self.value
    }

    pub fn unit(&self) -> Option<&'static str> {
        self.unit
    }
}

impl fmt::Display for TaskMetric {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.unit {
            Some(unit) => write!(formatter, "{}: {} {}", self.name, self.value, unit),
            None => write!(formatter, "{}: {}", self.name, self.value),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MetricValue {
    Integer(u64),
    Float(f64),
}

impl fmt::Display for MetricValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer(value) => write!(formatter, "{value}"),
            Self::Float(value) => write!(formatter, "{value:.3}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tasks::{Task, TaskMetric, TaskReport};

    #[test]
    fn writes_task_report_as_json() {
        let report = TaskReport::new(
            Task::LayoutThroughput,
            vec![
                TaskMetric::integer("samples", 128),
                TaskMetric::float("throughput", 42.5, Some("samples/s")),
            ],
        );
        let mut output = Vec::new();

        report.write_json(&mut output).unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("\"task\": \"layout-throughput\""));
        assert!(output.contains("\"name\": \"samples\""));
        assert!(output.contains("\"value\": 128"));
    }
}
