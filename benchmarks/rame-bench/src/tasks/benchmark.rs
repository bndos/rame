use crate::error::BenchResult;
use crate::tasks::{Task, TaskReport};

pub trait BenchmarkTask {
    type Model: ?Sized;

    fn name(&self) -> Task;

    fn evaluate(&self, model: &mut Self::Model) -> BenchResult<TaskReport>;
}
