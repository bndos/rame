use candle_core::D;

use crate::RameResult;
use crate::models::ModelError;
use crate::runtime::{JointNetwork, PredictionNetwork};
use crate::tensor::{Tensor, TensorError};
use crate::tokenization::TokenId;

use super::{TdtDecodingConfig, TdtJointOutput};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TdtStep {
    token_id: TokenId,
    duration: usize,
}

impl TdtStep {
    fn from_joint(output: &TdtJointOutput, config: &TdtDecodingConfig) -> RameResult<Self> {
        let token_id = argmax_single(output.token_logits(), "TDT token logits")?;
        let duration_index = argmax_single(output.duration_logits(), "TDT duration logits")?;
        let duration = config
            .durations()
            .get(duration_index)
            .copied()
            .ok_or_else(|| ModelError::InvalidTensorShape {
                name: "TDT duration logits".into(),
                expected: format!("last dimension {}", config.durations().len()),
                actual: output.duration_logits().dims().to_vec(),
            })?;

        Ok(Self {
            token_id: token_id as TokenId,
            duration,
        })
    }
}

fn argmax_single(logits: &Tensor, name: &'static str) -> RameResult<usize> {
    let indices = logits
        .argmax(D::Minus1)
        .and_then(|indices| indices.flatten_all())
        .and_then(|indices| indices.to_vec1::<u32>())
        .map_err(TensorError::from)?;

    if indices.len() != 1 {
        return Err(ModelError::InvalidTensorShape {
            name: name.into(),
            expected: "one logit vector".into(),
            actual: logits.dims().to_vec(),
        }
        .into());
    }

    Ok(indices[0] as usize)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdtTokenAlignment {
    token_id: TokenId,
    start_frame: usize,
    duration: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdtHypothesis {
    tokens: Vec<TdtTokenAlignment>,
}

impl TdtHypothesis {
    pub fn tokens(&self) -> &[TdtTokenAlignment] {
        &self.tokens
    }
}

impl TdtTokenAlignment {
    pub fn token_id(self) -> TokenId {
        self.token_id
    }

    pub fn start_frame(self) -> usize {
        self.start_frame
    }

    pub fn duration(self) -> usize {
        self.duration
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdtGreedySearch {
    config: TdtDecodingConfig,
}

impl TdtGreedySearch {
    pub fn new(config: TdtDecodingConfig) -> Self {
        Self { config }
    }

    pub fn decode<P, J>(
        &self,
        encoded: &J::Encoded,
        encoded_length: usize,
        predictor: &mut P,
        joint: &mut J,
    ) -> RameResult<TdtHypothesis>
    where
        P: PredictionNetwork<Token = TokenId>,
        J: JointNetwork<Predicted = P::Output, Output = TdtJointOutput>,
    {
        let mut hypothesis = TdtHypothesis { tokens: Vec::new() };
        let mut state = predictor.initial_state(1)?;
        let mut last_token = self.config.blank_token_id();
        let mut time = 0;

        while time < encoded_length {
            let mut duration = 0;

            for _ in 0..self.config.max_symbols_per_step() {
                let (predicted, candidate_state) = predictor.predict(&[last_token], &state)?;
                let output = joint.joint(encoded, &[time], &predicted)?;
                let next = TdtStep::from_joint(&output, &self.config)?;
                duration = next.duration;

                if next.token_id != self.config.blank_token_id() {
                    hypothesis.tokens.push(TdtTokenAlignment {
                        token_id: next.token_id,
                        start_frame: time,
                        duration,
                    });
                    last_token = next.token_id;
                    state = candidate_state;
                }

                time += duration;
                if duration != 0 {
                    break;
                }
            }

            if duration == 0 {
                time += 1;
            }
        }

        Ok(hypothesis)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::{TdtGreedySearch, TdtStep};
    use crate::RameResult;
    use crate::runtime::transducer::tdt::{TdtDecodingConfig, TdtJointLayout, TdtJointOutput};
    use crate::runtime::{JointNetwork, PredictionNetwork};
    use crate::tensor::Tensor;
    use crate::tokenization::TokenId;

    struct Predictor {
        initial_state: usize,
        calls: Vec<(TokenId, usize)>,
    }

    impl PredictionNetwork for Predictor {
        type Token = TokenId;
        type State = usize;
        type Output = ();

        fn initial_state(&mut self, batch_size: usize) -> RameResult<Self::State> {
            assert_eq!(batch_size, 1);
            Ok(self.initial_state)
        }

        fn predict(
            &mut self,
            tokens: &[Self::Token],
            state: &Self::State,
        ) -> RameResult<(Self::Output, Self::State)> {
            self.calls.push((tokens[0], *state));
            Ok(((), state + 1))
        }
    }

    struct Joint {
        layout: TdtJointLayout,
        steps: VecDeque<(TokenId, usize)>,
        visited_frames: Vec<usize>,
    }

    impl JointNetwork for Joint {
        type Encoded = ();
        type Predicted = ();
        type Output = TdtJointOutput;

        fn joint(
            &mut self,
            _encoded: &Self::Encoded,
            time_indices: &[usize],
            _predicted: &Self::Predicted,
        ) -> RameResult<Self::Output> {
            self.visited_frames.push(time_indices[0]);
            let (token_id, duration_index) = self.steps.pop_front().unwrap();
            Ok(joint_output(self.layout, token_id, duration_index))
        }
    }

    fn config(max_symbols_per_step: usize) -> TdtDecodingConfig {
        TdtDecodingConfig::new(8_193, 8_192, [0, 1, 2, 3, 4], max_symbols_per_step)
    }

    fn predictor(initial_state: usize) -> Predictor {
        Predictor {
            initial_state,
            calls: Vec::new(),
        }
    }

    fn joint(
        config: &TdtDecodingConfig,
        steps: impl IntoIterator<Item = (TokenId, usize)>,
    ) -> Joint {
        Joint {
            layout: TdtJointLayout::from(config),
            steps: steps.into_iter().collect(),
            visited_frames: Vec::new(),
        }
    }

    fn joint_output(
        layout: TdtJointLayout,
        token_id: TokenId,
        duration_index: usize,
    ) -> TdtJointOutput {
        let mut logits = vec![0.0f32; layout.joint_logit_count()];
        logits[token_id as usize] = 2.0;
        logits[layout.num_token_classes() + duration_index] = 4.0;
        TdtJointOutput::from_combined(
            Tensor::from_vec(logits, layout.joint_logit_count()).unwrap(),
            layout,
        )
        .unwrap()
    }

    #[test]
    fn selects_token_and_maps_duration_class_inside_decoding() {
        let config = config(10);
        let output = joint_output(TdtJointLayout::from(&config), 42, 3);

        assert_eq!(
            TdtStep::from_joint(&output, &config).unwrap(),
            TdtStep {
                token_id: 42,
                duration: 3,
            }
        );
    }

    #[test]
    fn repeats_zero_duration_steps_and_records_token_alignment() {
        let config = config(10);
        let blank_token_id = config.blank_token_id();
        let mut predictor = predictor(0);
        let mut joint = joint(&config, [(10, 0), (11, 2), (blank_token_id, 1)]);

        let hypothesis = TdtGreedySearch::new(config)
            .decode(&(), 3, &mut predictor, &mut joint)
            .unwrap();

        assert_eq!(hypothesis.tokens.len(), 2);
        assert_eq!(hypothesis.tokens[0].token_id, 10);
        assert_eq!(hypothesis.tokens[0].start_frame, 0);
        assert_eq!(hypothesis.tokens[0].duration, 0);
        assert_eq!(hypothesis.tokens[1].token_id, 11);
        assert_eq!(hypothesis.tokens[1].start_frame, 0);
        assert_eq!(hypothesis.tokens[1].duration, 2);
    }

    #[test]
    fn commits_prediction_state_only_when_a_token_is_emitted() {
        let config = config(10);
        let blank_token_id = config.blank_token_id();
        let mut predictor = predictor(7);
        let mut joint = joint(&config, [(blank_token_id, 1), (20, 1)]);

        TdtGreedySearch::new(config)
            .decode(&(), 2, &mut predictor, &mut joint)
            .unwrap();

        assert_eq!(predictor.calls, [(blank_token_id, 7), (blank_token_id, 7)]);
    }

    #[test]
    fn forces_frame_advance_after_the_symbol_limit() {
        let config = config(2);
        let blank_token_id = config.blank_token_id();
        let mut predictor = predictor(0);
        let mut joint = joint(&config, [(blank_token_id, 0); 4]);

        let hypothesis = TdtGreedySearch::new(config)
            .decode(&(), 2, &mut predictor, &mut joint)
            .unwrap();

        assert!(hypothesis.tokens.is_empty());
        assert_eq!(joint.visited_frames, [0, 0, 1, 1]);
    }
}
