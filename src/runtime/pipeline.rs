use std::marker::PhantomData;

use crate::RameResult;
use crate::runtime::{Decoder, Processor};
use crate::session::InferSession;

/// Typed composition of processing, inference, and decoding stages.
pub struct InferencePipeline<M, P, S, D> {
    architecture: PhantomData<M>,
    processor: P,
    session: S,
    decoder: D,
}

impl<M, P, S, D> InferencePipeline<M, P, S, D> {
    pub fn new(_architecture: M, processor: P, session: S, decoder: D) -> Self {
        Self {
            architecture: PhantomData,
            processor,
            session,
            decoder,
        }
    }
}

impl<M, P, S, D> InferencePipeline<M, P, S, D>
where
    P: Processor,
    S: InferSession,
    D: Decoder<Context = P::Context>,
{
    pub fn run(&mut self, source: &P::Source) -> RameResult<D::Output> {
        let processed = self.processor.process(source)?;
        let outputs = self.session.run(processed.inputs)?;
        self.decoder.decode(&outputs, &processed.context)
    }
}
