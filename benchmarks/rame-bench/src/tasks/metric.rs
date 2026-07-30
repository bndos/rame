use std::fmt;

use crate::tasks::Task;

#[derive(Debug, Clone, PartialEq)]
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
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, Copy, PartialEq)]
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
