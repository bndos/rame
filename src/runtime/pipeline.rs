use std::marker::PhantomData;

use crate::RameError;
use crate::RameResult;
use crate::runtime::{DecodeBatch, Decoder, Processor};
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
    pub fn run_many<'a>(&mut self, sources: &'a [P::Source<'a>]) -> RameResult<Vec<D::Output>> {
        crate::instrumentation::time_stage!("rame_pipeline_duration", self.run_many_inner(sources))
    }

    fn run_many_inner<'a>(&mut self, sources: &'a [P::Source<'a>]) -> RameResult<Vec<D::Output>> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }

        let source_len = sources.len();
        let processed = crate::instrumentation::time_stage!(
            "rame_pipeline_preprocess_duration",
            self.processor.process_many(sources)
        )?;
        if processed.len != source_len {
            return Err(RameError::InvalidBatchLength {
                stage: "processor output",
                expected: source_len,
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
            "rame_pipeline_inference_duration",
            self.session.run(processed.inputs)
        )?;
        let decoded = crate::instrumentation::time_stage!(
            "rame_pipeline_decode_duration",
            self.decoder.decode_batch(DecodeBatch {
                len: processed.len,
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
    }
}

#[cfg(test)]
mod tests {
    use crate::RameResult;
    use crate::runtime::{DecodeBatch, Decoder, InferencePipeline, ProcessedBatch, Processor};
    use crate::session::InferSession;
    use crate::tensor::TensorMap;

    #[derive(Debug, Clone, Copy)]
    struct TestArchitecture;

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
        runs: usize,
    }

    impl InferSession for CountingSession {
        fn run(&mut self, inputs: TensorMap) -> RameResult<TensorMap> {
            self.runs += 1;
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
    fn runs_batch_through_session_once() {
        let mut pipeline = InferencePipeline::new(
            TestArchitecture,
            EchoProcessor,
            CountingSession { runs: 0 },
            EchoDecoder,
        );

        let outputs = pipeline.run_many(&[1, 2, 3]).unwrap();

        assert_eq!(outputs, vec![2, 4, 6]);
        assert_eq!(pipeline.session.runs, 1);
    }
}
