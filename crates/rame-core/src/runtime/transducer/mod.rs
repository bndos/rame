pub mod tdt;

use crate::runtime::{ModelRunner, Processor};
use crate::tensor::Tensor;
use crate::{RameError, RameResult};

#[derive(Debug, Clone)]
pub struct TransducerEncoding<T = Tensor, L = Tensor> {
    encoded: T,
    lengths: L,
}

impl<T, L> TransducerEncoding<T, L> {
    pub fn new(encoded: T, lengths: L) -> Self {
        Self { encoded, lengths }
    }

    pub fn encoded(&self) -> &T {
        &self.encoded
    }

    pub fn lengths(&self) -> &L {
        &self.lengths
    }

    pub fn into_parts(self) -> (T, L) {
        (self.encoded, self.lengths)
    }
}

/// Converts preprocessed acoustic features into contextual representations.
pub trait TransducerEncoder {
    type Inputs;
    type Encoded;
    type Lengths;

    fn encode(
        &mut self,
        inputs: Self::Inputs,
    ) -> RameResult<TransducerEncoding<Self::Encoded, Self::Lengths>>;
}

/// Predicts a text-history representation from the last emitted tokens.
pub trait PredictionNetwork {
    type Tokens;
    type Output;
    type State;

    fn initial_state(&mut self, batch_size: usize) -> RameResult<Self::State>;

    fn predict(
        &mut self,
        tokens: &Self::Tokens,
        state: &Self::State,
    ) -> RameResult<(Self::Output, Self::State)>;
}

/// Commits candidate state only for rows that emitted a symbol.
pub trait MaskedPredictionNetwork: PredictionNetwork {
    type Mask;

    fn replace_prediction(
        &mut self,
        output: &mut Self::Output,
        state: &mut Self::State,
        candidate_output: Self::Output,
        candidate_state: Self::State,
        replace_mask: &Self::Mask,
    ) -> RameResult<()>;
}

/// Combines acoustic and text-history representations into output logits.
pub trait JointNetwork {
    type Encoded;
    type Indices;
    type Predicted;
    type Output;

    fn joint(
        &mut self,
        encoded: &Self::Encoded,
        time_indices: &Self::Indices,
        predicted: &Self::Predicted,
    ) -> RameResult<Self::Output>;
}

/// Controls transducer search and converts completed hypotheses into results.
pub trait TransducerDecoding<E, P, J>
where
    E: TransducerEncoder,
    P: PredictionNetwork,
    J: JointNetwork,
{
    type Context;
    type Output;

    fn decode_many(
        &mut self,
        encoding: TransducerEncoding<E::Encoded, E::Lengths>,
        contexts: &[Self::Context],
        predictor: &mut P,
        joint: &mut J,
    ) -> RameResult<Vec<Self::Output>>;
}

/// Processor -> encoder -> prediction/joint decoding model runner.
pub struct TransducerModelRunner<A, E, P, J, D> {
    preprocessor: A,
    encoder: E,
    predictor: P,
    joint: J,
    decoding: D,
}

impl<A, E, P, J, D> TransducerModelRunner<A, E, P, J, D>
where
    A: Processor,
    E: TransducerEncoder<Inputs = A::Inputs>,
    P: PredictionNetwork,
    J: JointNetwork,
    D: TransducerDecoding<E, P, J, Context = A::Context>,
{
    pub fn new(preprocessor: A, encoder: E, predictor: P, joint: J, decoding: D) -> Self {
        Self {
            preprocessor,
            encoder,
            predictor,
            joint,
            decoding,
        }
    }

    pub fn run<'a>(&mut self, sources: &'a [A::Source<'a>]) -> RameResult<Vec<D::Output>> {
        crate::instrumentation::time_stage!(
            "rame_model_runner_duration",
            (|| {
                if sources.is_empty() {
                    return Ok(Vec::new());
                }

                let processed = crate::instrumentation::time_stage!(
                    "rame_model_runner_preprocess_duration",
                    self.preprocessor.process_many(sources)
                )?;
                validate_batch_length("processor output", sources.len(), processed.len)?;
                validate_batch_length(
                    "processor contexts",
                    processed.len,
                    processed.contexts.len(),
                )?;

                let encoded = crate::instrumentation::time_stage!(
                    "rame_model_runner_encode_duration",
                    self.encoder.encode(processed.inputs)
                )?;

                let outputs = crate::instrumentation::time_stage!(
                    "rame_model_runner_decode_duration",
                    self.decoding.decode_many(
                        encoded,
                        &processed.contexts,
                        &mut self.predictor,
                        &mut self.joint,
                    )
                )?;
                validate_batch_length("decoding output", processed.len, outputs.len())?;

                Ok(outputs)
            })()
        )
    }
}

impl<A, E, P, J, D> ModelRunner for TransducerModelRunner<A, E, P, J, D>
where
    A: Processor,
    E: TransducerEncoder<Inputs = A::Inputs>,
    P: PredictionNetwork,
    J: JointNetwork,
    D: TransducerDecoding<E, P, J, Context = A::Context>,
{
    type Input<'a> = A::Source<'a>;
    type Output = D::Output;

    fn run_many<'a>(&mut self, inputs: &'a [Self::Input<'a>]) -> RameResult<Vec<Self::Output>> {
        self.run(inputs)
    }
}

fn validate_batch_length(stage: &'static str, expected: usize, actual: usize) -> RameResult<()> {
    if actual != expected {
        return Err(RameError::InvalidBatchLength {
            stage,
            expected,
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::RameResult;
    use crate::runtime::{
        JointNetwork, MaskedPredictionNetwork, PredictionNetwork, ProcessedBatch, Processor,
        TransducerDecoding, TransducerEncoder,
    };
    use crate::tensor::{Device, Tensor, TensorMap};

    use super::TransducerModelRunner;

    struct ContextProcessor;

    impl Processor for ContextProcessor {
        type Source<'a> = usize;
        type Context = usize;
        type Inputs = TensorMap;

        fn process_many<'a>(
            &self,
            sources: &'a [Self::Source<'a>],
        ) -> RameResult<ProcessedBatch<Self::Context>> {
            Ok(ProcessedBatch {
                len: sources.len(),
                inputs: TensorMap::new(),
                contexts: sources.to_vec(),
            })
        }
    }

    struct Encoder;

    impl TransducerEncoder for Encoder {
        type Inputs = TensorMap;
        type Encoded = Tensor;
        type Lengths = Tensor;

        fn encode(&mut self, _inputs: TensorMap) -> RameResult<super::TransducerEncoding> {
            Ok(super::TransducerEncoding::new(
                Tensor::zeros((3, 1, 1), crate::tensor::DType::F32, &Device::Cpu).unwrap(),
                Tensor::new(&[1u32, 1, 1], &Device::Cpu).unwrap(),
            ))
        }
    }

    struct Predictor;

    impl PredictionNetwork for Predictor {
        type Tokens = Tensor;
        type Output = Tensor;
        type State = ();

        fn initial_state(&mut self, _batch_size: usize) -> RameResult<Self::State> {
            Ok(())
        }

        fn predict(
            &mut self,
            _tokens: &Tensor,
            _state: &Self::State,
        ) -> RameResult<(Tensor, Self::State)> {
            Ok((
                Tensor::zeros((3, 1), crate::tensor::DType::F32, &Device::Cpu).unwrap(),
                (),
            ))
        }
    }

    impl MaskedPredictionNetwork for Predictor {
        type Mask = Tensor;
        fn replace_prediction(
            &mut self,
            _output: &mut Tensor,
            _state: &mut Self::State,
            _candidate_output: Tensor,
            _candidate_state: Self::State,
            _replace_mask: &Tensor,
        ) -> RameResult<()> {
            Ok(())
        }
    }

    struct Joint;

    impl JointNetwork for Joint {
        type Encoded = Tensor;
        type Indices = Tensor;
        type Predicted = Tensor;
        type Output = ();

        fn joint(
            &mut self,
            _encoded: &Tensor,
            _time_indices: &Tensor,
            _predicted: &Tensor,
        ) -> RameResult<Self::Output> {
            Ok(())
        }
    }

    struct ContextDecoding;

    impl TransducerDecoding<Encoder, Predictor, Joint> for ContextDecoding {
        type Context = usize;
        type Output = usize;

        fn decode_many(
            &mut self,
            _encoding: super::TransducerEncoding,
            contexts: &[Self::Context],
            _predictor: &mut Predictor,
            _joint: &mut Joint,
        ) -> RameResult<Vec<Self::Output>> {
            Ok(contexts.iter().map(|context| context * 2).collect())
        }
    }

    #[test]
    fn carries_processor_contexts_into_transducer_decoding() {
        let mut runner = TransducerModelRunner::new(
            ContextProcessor,
            Encoder,
            Predictor,
            Joint,
            ContextDecoding,
        );

        assert_eq!(runner.run(&[1, 2, 3]).unwrap(), [2, 4, 6]);
    }
}
