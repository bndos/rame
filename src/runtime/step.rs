use crate::RameResult;

/// One typed operation in a runtime pipeline.
pub trait PipelineStep<Input> {
    type Output;

    fn execute(&mut self, input: Input) -> RameResult<Self::Output>;
}
