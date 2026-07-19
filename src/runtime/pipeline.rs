use crate::RameResult;
use crate::runtime::{Decoder, Processor};
use crate::session::InferSession;

/// Typed composition of processing, inference, and decoding stages.
pub struct InferencePipeline<P, S, D> {
    processor: P,
    session: S,
    decoder: D,
}

impl<P, S, D> InferencePipeline<P, S, D> {
    pub fn new(processor: P, session: S, decoder: D) -> Self {
        Self {
            processor,
            session,
            decoder,
        }
    }
}

impl<P, S, D> InferencePipeline<P, S, D>
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
