use crate::RameResult;

/// Executes one loaded semantic model.
pub trait ModelRunner {
    type Input<'a>;
    type Output;

    fn run_many<'a>(&mut self, inputs: &'a [Self::Input<'a>]) -> RameResult<Vec<Self::Output>>;
}
