use crate::runtime::{ModelRunner, Processor};
use crate::tensor::TensorMap;
use crate::{RameError, RameResult};

/// Converts preprocessed acoustic features into contextual representations.
pub trait TransducerEncoder {
    type Output;

    fn encode(&mut self, inputs: TensorMap) -> RameResult<Self::Output>;
}

/// Predicts a text-history representation from the last emitted tokens.
pub trait PredictionNetwork {
    type Token;
    type State;
    type Output;

    fn initial_state(&mut self, batch_size: usize) -> RameResult<Self::State>;

    fn predict(
        &mut self,
        tokens: &[Self::Token],
        state: &Self::State,
    ) -> RameResult<(Self::Output, Self::State)>;
}

/// Combines acoustic and text-history representations into output logits.
pub trait JointNetwork {
    type Encoded;
    type Predicted;
    type Output;

    fn joint(
        &mut self,
        encoded: &Self::Encoded,
        time_indices: &[usize],
        predicted: &Self::Predicted,
    ) -> RameResult<Self::Output>;
}

/// Controls transducer search and converts completed hypotheses into results.
pub trait TransducerDecoding<E, P, J>
where
    E: TransducerEncoder,
    P: PredictionNetwork,
    J: JointNetwork<Encoded = E::Output, Predicted = P::Output>,
{
    type Output;

    fn decode_many(
        &mut self,
        encoded: E::Output,
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
    E: TransducerEncoder,
    P: PredictionNetwork,
    J: JointNetwork<Encoded = E::Output, Predicted = P::Output>,
    D: TransducerDecoding<E, P, J>,
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

                let encoded = crate::instrumentation::time_stage!(
                    "rame_model_runner_encode_duration",
                    self.encoder.encode(processed.inputs)
                )?;

                let outputs = crate::instrumentation::time_stage!(
                    "rame_model_runner_decode_duration",
                    self.decoding
                        .decode_many(encoded, &mut self.predictor, &mut self.joint)
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
    E: TransducerEncoder,
    P: PredictionNetwork,
    J: JointNetwork<Encoded = E::Output, Predicted = P::Output>,
    D: TransducerDecoding<E, P, J>,
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
