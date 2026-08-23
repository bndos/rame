use crate::RameResult;
use crate::runtime::PipelineStep;

/// Typed runtime pipeline composed from one or more steps.
pub struct Pipeline<S> {
    step: S,
}

impl<S> Pipeline<S> {
    pub fn new(step: S) -> Self {
        Self { step }
    }

    pub fn then<N>(self, next: N) -> Pipeline<Then<S, N>> {
        Pipeline::new(Then {
            first: self.step,
            second: next,
        })
    }

    pub fn run<Input>(&mut self, input: Input) -> RameResult<S::Output>
    where
        S: PipelineStep<Input>,
    {
        crate::instrumentation::time_stage!("rame_pipeline_duration", self.step.execute(input))
    }
}

impl<Input, S> PipelineStep<Input> for Pipeline<S>
where
    S: PipelineStep<Input>,
{
    type Output = S::Output;

    fn execute(&mut self, input: Input) -> RameResult<Self::Output> {
        self.step.execute(input)
    }
}

/// Two pipeline steps executed sequentially.
pub struct Then<A, B> {
    first: A,
    second: B,
}

impl<Input, A, B> PipelineStep<Input> for Then<A, B>
where
    A: PipelineStep<Input>,
    B: PipelineStep<A::Output>,
{
    type Output = B::Output;

    fn execute(&mut self, input: Input) -> RameResult<Self::Output> {
        let intermediate = self.first.execute(input)?;
        self.second.execute(intermediate)
    }
}

#[cfg(test)]
mod tests {
    use crate::RameResult;
    use crate::runtime::{Pipeline, PipelineStep};

    struct AddOne;

    impl PipelineStep<u32> for AddOne {
        type Output = u64;

        fn execute(&mut self, input: u32) -> RameResult<Self::Output> {
            Ok(u64::from(input) + 1)
        }
    }

    struct ToString;

    impl PipelineStep<u64> for ToString {
        type Output = String;

        fn execute(&mut self, input: u64) -> RameResult<Self::Output> {
            Ok(input.to_string())
        }
    }

    #[test]
    fn composes_steps_with_different_types() {
        let mut pipeline = Pipeline::new(AddOne).then(ToString);

        assert_eq!(pipeline.run(41).unwrap(), "42");
    }

    #[test]
    fn composes_nested_pipelines() {
        let stringify = Pipeline::new(ToString);
        let mut pipeline = Pipeline::new(AddOne).then(stringify);

        assert_eq!(pipeline.run(41).unwrap(), "42");
    }
}
