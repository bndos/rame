use std::marker::PhantomData;

use crate::RameError;
use crate::RameResult;
use crate::runtime::{DecodeBatch, Decoder, Pipeline, PipelineStep, ProcessedBatch, Processor};
use crate::session::InferSession;
use crate::tensor::TensorMap;

/// Preprocessing step for one model architecture.
struct ProcessStep<M, P> {
    architecture: PhantomData<M>,
    processor: P,
}

impl<M, P> ProcessStep<M, P> {
    pub fn new(_architecture: M, processor: P) -> Self {
        Self {
            architecture: PhantomData,
            processor,
        }
    }
}

impl<'a, M, P> PipelineStep<&'a [P::Source<'a>]> for ProcessStep<M, P>
where
    P: Processor,
{
    type Output = ProcessedBatch<P::Context>;

    fn execute(&mut self, sources: &'a [P::Source<'a>]) -> RameResult<Self::Output> {
        if sources.is_empty() {
            return Ok(ProcessedBatch {
                len: 0,
                inputs: TensorMap::new(),
                contexts: Vec::new(),
            });
        }

        let processed = crate::instrumentation::time_stage!(
            "rame_pipeline_preprocess_duration",
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
        Ok(processed)
    }
}

/// Inference outputs paired with the contexts required for decoding.
struct InferredBatch<C> {
    len: usize,
    outputs: TensorMap,
    contexts: Vec<C>,
}

/// Model-session execution step.
struct InferenceStep<S> {
    session: S,
}

impl<S> InferenceStep<S> {
    pub fn new(session: S) -> Self {
        Self { session }
    }
}

impl<C, S> PipelineStep<ProcessedBatch<C>> for InferenceStep<S>
where
    S: InferSession,
{
    type Output = InferredBatch<C>;

    fn execute(&mut self, batch: ProcessedBatch<C>) -> RameResult<Self::Output> {
        let outputs = if batch.len == 0 {
            TensorMap::new()
        } else {
            crate::instrumentation::time_stage!(
                "rame_pipeline_inference_duration",
                self.session.run(batch.inputs)
            )?
        };
        Ok(InferredBatch {
            len: batch.len,
            outputs,
            contexts: batch.contexts,
        })
    }
}

/// Typed output-decoding step.
struct DecodeStep<D> {
    decoder: D,
}

impl<D> DecodeStep<D> {
    pub fn new(decoder: D) -> Self {
        Self { decoder }
    }
}

impl<C, D> PipelineStep<InferredBatch<C>> for DecodeStep<D>
where
    D: Decoder<Context = C>,
{
    type Output = Vec<D::Output>;

    fn execute(&mut self, batch: InferredBatch<C>) -> RameResult<Self::Output> {
        if batch.len == 0 {
            return Ok(Vec::new());
        }

        let decoded = crate::instrumentation::time_stage!(
            "rame_pipeline_decode_duration",
            self.decoder.decode_batch(DecodeBatch {
                len: batch.len,
                outputs: &batch.outputs,
                contexts: &batch.contexts,
            })
        )?;
        if decoded.len() != batch.len {
            return Err(RameError::InvalidBatchLength {
                stage: "decoder output",
                expected: batch.len,
                actual: decoded.len(),
            });
        }
        Ok(decoded)
    }
}

type ModelPipelineInner<M, P, S, D> = Pipeline<
    crate::runtime::pipeline::Then<
        crate::runtime::pipeline::Then<ProcessStep<M, P>, InferenceStep<S>>,
        DecodeStep<D>,
    >,
>;

pub struct ModelPipeline<M, P, S, D> {
    inner: ModelPipelineInner<M, P, S, D>,
}

impl<M, P, S, D> ModelPipeline<M, P, S, D>
where
    P: Processor,
    S: InferSession,
    D: Decoder<Context = P::Context>,
{
    pub fn new(_architecture: M, processor: P, session: S, decoder: D) -> Self {
        Self {
            inner: Pipeline::new(ProcessStep::new(_architecture, processor))
                .then(InferenceStep::new(session))
                .then(DecodeStep::new(decoder)),
        }
    }

    pub fn run<'a>(&mut self, sources: &'a [P::Source<'a>]) -> RameResult<Vec<D::Output>> {
        self.inner.run(sources)
    }
}

impl<'a, M, P, S, D> PipelineStep<&'a [P::Source<'a>]> for ModelPipeline<M, P, S, D>
where
    P: Processor,
    S: InferSession,
    D: Decoder<Context = P::Context>,
{
    type Output = Vec<D::Output>;

    fn execute(&mut self, sources: &'a [P::Source<'a>]) -> RameResult<Self::Output> {
        self.run(sources)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use crate::RameResult;
    use crate::runtime::{DecodeBatch, Decoder, Pipeline, PipelineStep, ProcessedBatch, Processor};
    use crate::session::InferSession;
    use crate::tensor::TensorMap;

    use super::{DecodeStep, InferenceStep, ModelPipeline};

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
    fn runs_standard_model_pipeline_once() {
        let runs = Rc::new(Cell::new(0));
        let mut pipeline = ModelPipeline::new(
            TestArchitecture,
            EchoProcessor,
            CountingSession {
                runs: Rc::clone(&runs),
            },
            EchoDecoder,
        );

        let outputs = pipeline.run(&[1, 2, 3][..]).unwrap();

        assert_eq!(outputs, vec![2, 4, 6]);
        assert_eq!(runs.get(), 1);
    }

    #[test]
    fn skips_inference_for_empty_batches() {
        let runs = Rc::new(Cell::new(0));
        let mut pipeline = ModelPipeline::new(
            TestArchitecture,
            EchoProcessor,
            CountingSession {
                runs: Rc::clone(&runs),
            },
            EchoDecoder,
        );

        let inputs: &[i32] = &[];
        let outputs = pipeline.run(inputs).unwrap();

        assert!(outputs.is_empty());
        assert_eq!(runs.get(), 0);
    }

    struct Reprocess;

    impl PipelineStep<Vec<i32>> for Reprocess {
        type Output = ProcessedBatch<i32>;

        fn execute(&mut self, values: Vec<i32>) -> RameResult<Self::Output> {
            Ok(ProcessedBatch {
                len: values.len(),
                inputs: TensorMap::new(),
                contexts: values,
            })
        }
    }

    #[test]
    fn composes_multiple_model_executions() {
        let first = ModelPipeline::new(
            TestArchitecture,
            EchoProcessor,
            CountingSession {
                runs: Rc::new(Cell::new(0)),
            },
            EchoDecoder,
        );
        let second = Pipeline::new(Reprocess)
            .then(InferenceStep::new(CountingSession {
                runs: Rc::new(Cell::new(0)),
            }))
            .then(DecodeStep::new(EchoDecoder));
        let mut pipeline = Pipeline::new(first).then(second);

        let outputs = pipeline.run(&[1, 2, 3][..]).unwrap();

        assert_eq!(outputs, vec![4, 8, 12]);
    }
}
