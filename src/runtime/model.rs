use crate::runtime::{DecodeBatch, Decoder, Processor};
use crate::session::InferSession;
use crate::{RameError, RameResult};

/// Executes one loaded semantic model.
///
/// A runner owns the runtime resources and control flow needed to complete a
/// batch. Autoregressive models may implement this trait with a stateful loop.
/// single-session models can use [`StandardModelRunner`].
pub trait ModelRunner {
    type Input<'a>;
    type Output;

    fn run_many<'a>(&mut self, inputs: &'a [Self::Input<'a>]) -> RameResult<Vec<Self::Output>>;
}

/// Standard processor -> session -> decoder model runner.
pub struct StandardModelRunner<P, S, D> {
    processor: P,
    session: S,
    decoder: D,
}

impl<P, S, D> StandardModelRunner<P, S, D>
where
    P: Processor,
    S: InferSession,
    D: Decoder<Context = P::Context>,
{
    pub fn new(processor: P, session: S, decoder: D) -> Self {
        Self {
            processor,
            session,
            decoder,
        }
    }

    pub fn run<'a>(&mut self, sources: &'a [P::Source<'a>]) -> RameResult<Vec<D::Output>> {
        crate::instrumentation::time_stage!(
            "rame_model_runner_duration",
            (|| {
                if sources.is_empty() {
                    return Ok(Vec::new());
                }

                let processed = crate::instrumentation::time_stage!(
                    "rame_model_runner_preprocess_duration",
                    self.processor.process_many(sources)
                )?;
                if processed.len != sources.len() {
                    return Err(RameError::InvalidBatchLength {
                        stage: "processor output",
                        expected: sources.len(),
                        actual: processed.len,
                    });
                }
                if processed.contexts.len() != processed.len {
                    return Err(RameError::InvalidBatchLength {
                        stage: "processor contexts",
                        expected: processed.len,
                        actual: processed.contexts.len(),
                    });
                }

                let outputs = crate::instrumentation::time_stage!(
                    "rame_model_runner_inference_duration",
                    self.session.run(processed.inputs)
                )?;
                let decoded = crate::instrumentation::time_stage!(
                    "rame_model_runner_decode_duration",
                    self.decoder.decode_batch(DecodeBatch {
                        outputs: &outputs,
                        contexts: &processed.contexts,
                    })
                )?;
                if decoded.len() != processed.len {
                    return Err(RameError::InvalidBatchLength {
                        stage: "decoder output",
                        expected: processed.len,
                        actual: decoded.len(),
                    });
                }

                Ok(decoded)
            })()
        )
    }
}

impl<P, S, D> ModelRunner for StandardModelRunner<P, S, D>
where
    P: Processor,
    S: InferSession,
    D: Decoder<Context = P::Context>,
{
    type Input<'a> = P::Source<'a>;
    type Output = D::Output;

    fn run_many<'a>(&mut self, inputs: &'a [Self::Input<'a>]) -> RameResult<Vec<Self::Output>> {
        self.run(inputs)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use crate::RameResult;
    use crate::runtime::{DecodeBatch, Decoder, ModelRunner, ProcessedBatch, Processor};
    use crate::session::InferSession;
    use crate::tensor::TensorMap;

    use super::StandardModelRunner;

    struct EchoProcessor;

    impl Processor for EchoProcessor {
        type Source<'a> = i32;
        type Context = i32;

        fn process_many<'a>(
            &self,
            sources: &'a [Self::Source<'a>],
        ) -> RameResult<ProcessedBatch<i32>> {
            Ok(ProcessedBatch {
                len: sources.len(),
                inputs: TensorMap::new(),
                contexts: sources.to_vec(),
            })
        }
    }

    struct CountingSession {
        runs: Rc<Cell<usize>>,
    }

    impl InferSession for CountingSession {
        fn run(&mut self, inputs: TensorMap) -> RameResult<TensorMap> {
            self.runs.set(self.runs.get() + 1);
            Ok(inputs)
        }
    }

    struct EchoDecoder;

    impl Decoder for EchoDecoder {
        type Output = i32;
        type Context = i32;

        fn decode_batch(
            &self,
            batch: DecodeBatch<'_, Self::Context>,
        ) -> RameResult<Vec<Self::Output>> {
            Ok(batch.contexts.iter().map(|value| value * 2).collect())
        }
    }

    #[test]
    fn runs_standard_model_once() {
        let runs = Rc::new(Cell::new(0));
        let mut runner = StandardModelRunner::new(
            EchoProcessor,
            CountingSession {
                runs: Rc::clone(&runs),
            },
            EchoDecoder,
        );

        let outputs = runner.run_many(&[1, 2, 3]).unwrap();

        assert_eq!(outputs, vec![2, 4, 6]);
        assert_eq!(runs.get(), 1);
    }

    #[test]
    fn skips_standard_model_for_empty_batches() {
        let runs = Rc::new(Cell::new(0));
        let mut runner = StandardModelRunner::new(
            EchoProcessor,
            CountingSession {
                runs: Rc::clone(&runs),
            },
            EchoDecoder,
        );

        assert!(runner.run_many(&[]).unwrap().is_empty());
        assert_eq!(runs.get(), 0);
    }
}
