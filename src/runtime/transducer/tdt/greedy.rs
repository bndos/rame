use crate::RameResult;
use crate::models::ModelError;
use crate::runtime::{JointNetwork, PredictionNetwork, expect_one};
use crate::tensor::{D, Tensor, TensorError};
use crate::tokenization::TokenId;

use super::{TdtDecodingConfig, TdtHypothesis, TdtJointOutput};

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
        let hypotheses = self.decode_many(encoded, &[encoded_length], predictor, joint)?;
        expect_one(hypotheses, "TDT greedy hypothesis")
    }

    pub fn decode_many<P, J>(
        &self,
        encoded: &J::Encoded,
        encoded_lengths: &[usize],
        predictor: &mut P,
        joint: &mut J,
    ) -> RameResult<Vec<TdtHypothesis>>
    where
        P: PredictionNetwork<Token = TokenId>,
        J: JointNetwork<Predicted = P::Output, Output = TdtJointOutput>,
    {
        let batch_size = encoded_lengths.len();
        let mut hypotheses = (0..batch_size)
            .map(|_| TdtHypothesis::new())
            .collect::<Vec<_>>();
        if batch_size == 0 || encoded_lengths.iter().all(|&length| length == 0) {
            return Ok(hypotheses);
        }

        let blank_token_id = self.config.blank_token_id();
        let initial_state = predictor.initial_state(batch_size)?;
        let start_tokens = vec![blank_token_id; batch_size];
        let (mut predicted, mut state) = predictor.predict(&start_tokens, &initial_state)?;
        let mut time_indices = vec![0; batch_size];
        let mut last_symbol_times = vec![None; batch_size];
        let mut symbols_at_time = vec![0; batch_size];
        let mut active = vec![false; batch_size];
        let mut seeking_label = vec![false; batch_size];
        let mut labels = vec![blank_token_id; batch_size];
        let mut label_times = vec![0; batch_size];
        let mut label_durations = vec![0; batch_size];
        let mut found_label = vec![false; batch_size];
        let mut safe_time_indices = vec![0; batch_size];

        while any_active(&time_indices, encoded_lengths) {
            for batch_index in 0..batch_size {
                active[batch_index] = time_indices[batch_index] < encoded_lengths[batch_index];
            }
            seeking_label.clone_from(&active);
            labels.fill(blank_token_id);
            label_times.fill(0);
            label_durations.fill(0);

            while seeking_label.iter().any(|&seeking| seeking) {
                for batch_index in 0..batch_size {
                    safe_time_indices[batch_index] = time_indices[batch_index]
                        .min(encoded_lengths[batch_index].saturating_sub(1));
                }
                let output = joint.joint(encoded, &safe_time_indices, &predicted)?;
                let (token_ids, durations) = select_batch(&output, &self.config, batch_size)?;

                for batch_index in 0..batch_size {
                    if !seeking_label[batch_index] {
                        continue;
                    }

                    let token_id = token_ids[batch_index];
                    let duration = durations[batch_index];
                    label_times[batch_index] = time_indices[batch_index];
                    label_durations[batch_index] = duration;

                    if token_id == blank_token_id {
                        time_indices[batch_index] =
                            time_indices[batch_index].saturating_add(duration.max(1));
                        seeking_label[batch_index] =
                            time_indices[batch_index] < encoded_lengths[batch_index];
                    } else {
                        labels[batch_index] = token_id;
                        time_indices[batch_index] =
                            time_indices[batch_index].saturating_add(duration);
                        seeking_label[batch_index] = false;
                    }
                }
            }

            let mut any_found_label = false;
            for batch_index in 0..batch_size {
                found_label[batch_index] =
                    active[batch_index] && labels[batch_index] != blank_token_id;

                if !found_label[batch_index] {
                    continue;
                }
                any_found_label = true;

                let label_time = label_times[batch_index];
                let duration = label_durations[batch_index];
                hypotheses[batch_index].push(labels[batch_index], label_time, duration);

                if last_symbol_times[batch_index] == Some(label_time) {
                    symbols_at_time[batch_index] += 1;
                } else {
                    last_symbol_times[batch_index] = Some(label_time);
                    symbols_at_time[batch_index] = 1;
                }

                if time_indices[batch_index] < encoded_lengths[batch_index]
                    && time_indices[batch_index] == label_time
                    && symbols_at_time[batch_index] >= self.config.max_symbols_per_step()
                {
                    time_indices[batch_index] += 1;
                }
            }

            if any_found_label {
                let (candidate_output, candidate_state) = predictor.predict(&labels, &state)?;
                predictor.replace_prediction(
                    &mut predicted,
                    &mut state,
                    candidate_output,
                    candidate_state,
                    &found_label,
                )?;
            }
        }

        Ok(hypotheses)
    }
}

fn any_active(time_indices: &[usize], encoded_lengths: &[usize]) -> bool {
    time_indices
        .iter()
        .zip(encoded_lengths)
        .any(|(&time, &length)| time < length)
}

fn select_batch(
    output: &TdtJointOutput,
    config: &TdtDecodingConfig,
    batch_size: usize,
) -> RameResult<(Vec<TokenId>, Vec<usize>)> {
    let token_ids = argmax_batch(output.token_logits(), "TDT token logits", batch_size)?;
    let duration_indices =
        argmax_batch(output.duration_logits(), "TDT duration logits", batch_size)?;
    let durations = duration_indices
        .into_iter()
        .map(|duration_index| {
            config
                .durations()
                .get(duration_index)
                .copied()
                .ok_or_else(|| ModelError::InvalidTensorShape {
                    name: "TDT duration logits".into(),
                    expected: format!("last dimension {}", config.durations().len()),
                    actual: output.duration_logits().dims().to_vec(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((
        token_ids
            .into_iter()
            .map(|token_id| token_id as TokenId)
            .collect(),
        durations,
    ))
}

fn argmax_batch(logits: &Tensor, name: &'static str, batch_size: usize) -> RameResult<Vec<usize>> {
    let indices = logits
        .argmax(D::Minus1)
        .and_then(|indices| indices.flatten_all())
        .and_then(|indices| indices.to_vec1::<u32>())
        .map_err(TensorError::from)?;

    if indices.len() != batch_size {
        return Err(ModelError::InvalidTensorShape {
            name: name.into(),
            expected: format!("{batch_size} logit vectors"),
            actual: logits.dims().to_vec(),
        }
        .into());
    }

    Ok(indices.into_iter().map(|index| index as usize).collect())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crate::RameResult;
    use crate::runtime::transducer::tdt::{
        TdtDecodingConfig, TdtGreedySearch, TdtJointLayout, TdtJointOutput,
    };
    use crate::runtime::{JointNetwork, PredictionNetwork};
    use crate::tensor::Tensor;
    use crate::tokenization::TokenId;

    struct Predictor {
        calls: Vec<Vec<TokenId>>,
        output: Vec<TokenId>,
        state: Vec<usize>,
    }

    impl PredictionNetwork for Predictor {
        type Token = TokenId;
        type State = Vec<usize>;
        type Output = Vec<TokenId>;

        fn initial_state(&mut self, batch_size: usize) -> RameResult<Self::State> {
            Ok(vec![0; batch_size])
        }

        fn predict(
            &mut self,
            tokens: &[Self::Token],
            state: &Self::State,
        ) -> RameResult<(Self::Output, Self::State)> {
            self.calls.push(tokens.to_vec());
            Ok((
                tokens.to_vec(),
                state.iter().map(|value| value + 1).collect(),
            ))
        }

        fn replace_prediction(
            &mut self,
            output: &mut Self::Output,
            state: &mut Self::State,
            candidate_output: Self::Output,
            candidate_state: Self::State,
            replace_mask: &[bool],
        ) -> RameResult<()> {
            for batch_index in 0..replace_mask.len() {
                if replace_mask[batch_index] {
                    output[batch_index] = candidate_output[batch_index];
                    state[batch_index] = candidate_state[batch_index];
                }
            }
            self.output.clone_from(output);
            self.state.clone_from(state);
            Ok(())
        }
    }

    struct Joint {
        layout: TdtJointLayout,
        steps: VecDeque<Vec<(TokenId, usize)>>,
        time_indices: Vec<Vec<usize>>,
        predictions: Vec<Vec<TokenId>>,
    }

    impl JointNetwork for Joint {
        type Encoded = ();
        type Predicted = Vec<TokenId>;
        type Output = TdtJointOutput;

        fn joint(
            &mut self,
            _encoded: &Self::Encoded,
            time_indices: &[usize],
            predicted: &Self::Predicted,
        ) -> RameResult<Self::Output> {
            self.time_indices.push(time_indices.to_vec());
            self.predictions.push(predicted.clone());
            let steps = self.steps.pop_front().unwrap();
            let mut logits = vec![0.0f32; steps.len() * self.layout.joint_logit_count()];
            for (batch_index, (token_id, duration_index)) in steps.into_iter().enumerate() {
                let offset = batch_index * self.layout.joint_logit_count();
                logits[offset + token_id as usize] = 2.0;
                logits[offset + self.layout.num_token_classes() + duration_index] = 4.0;
            }
            TdtJointOutput::from_combined(
                Tensor::from_vec(
                    logits,
                    (time_indices.len(), self.layout.joint_logit_count()),
                    &crate::tensor::Device::Cpu,
                )
                .map_err(crate::tensor::TensorError::from)?,
                self.layout,
            )
        }
    }

    fn joint(layout: TdtJointLayout, steps: Vec<Vec<(TokenId, usize)>>) -> Joint {
        Joint {
            layout,
            steps: steps.into(),
            time_indices: Vec::new(),
            predictions: Vec::new(),
        }
    }

    fn predictor() -> Predictor {
        Predictor {
            calls: Vec::new(),
            output: Vec::new(),
            state: Vec::new(),
        }
    }

    #[test]
    fn loops_over_blanks_without_recomputing_the_prediction_network() {
        let config = TdtDecodingConfig::new(8_193, 8_192, [0, 1, 2, 3, 4], 10);
        let blank = config.blank_token_id();
        let layout = TdtJointLayout::from(&config);
        let mut predictor = predictor();
        let mut joint = joint(
            layout,
            vec![
                vec![(10, 0), (blank, 1)],
                vec![(blank, 4), (20, 1)],
                vec![(11, 2), (blank, 0)],
                vec![(blank, 1), (blank, 0)],
            ],
        );

        let hypotheses = TdtGreedySearch::new(config)
            .decode_many(&(), &[3, 2], &mut predictor, &mut joint)
            .unwrap();

        assert_eq!(
            hypotheses[0]
                .tokens()
                .iter()
                .map(|token| (token.token_id(), token.start_frame(), token.duration()))
                .collect::<Vec<_>>(),
            [(10, 0, 0), (11, 0, 2)]
        );
        assert_eq!(
            hypotheses[1]
                .tokens()
                .iter()
                .map(|token| (token.token_id(), token.start_frame(), token.duration()))
                .collect::<Vec<_>>(),
            [(20, 1, 1)]
        );
        assert_eq!(predictor.calls, [[blank, blank], [10, 20], [11, blank]]);
        assert_eq!(predictor.output, [11, 20]);
        assert_eq!(predictor.state, [3, 2]);
        assert_eq!(joint.time_indices, [[0, 0], [0, 1], [0, 1], [2, 1]]);
        assert_eq!(
            joint.predictions,
            [
                vec![blank, blank],
                vec![blank, blank],
                vec![10, 20],
                vec![11, 20]
            ]
        );
    }

    #[test]
    fn decodes_one_through_the_batched_path() {
        let config = TdtDecodingConfig::new(8_193, 8_192, [0, 1, 2, 3, 4], 10);
        let blank = config.blank_token_id();
        let layout = TdtJointLayout::from(&config);
        let mut predictor = predictor();
        let mut joint = joint(layout, vec![vec![(10, 1)], vec![(blank, 1)]]);

        let hypothesis = TdtGreedySearch::new(config)
            .decode(&(), 2, &mut predictor, &mut joint)
            .unwrap();

        assert_eq!(hypothesis.tokens().len(), 1);
        assert_eq!(hypothesis.tokens()[0].token_id(), 10);
        assert_eq!(predictor.calls, [[blank], [10]]);
        assert_eq!(joint.time_indices, [[0], [1]]);
    }

    #[test]
    fn forces_zero_duration_blanks_to_advance_without_counting_symbols() {
        let config = TdtDecodingConfig::new(8_193, 8_192, [0, 1, 2, 3, 4], 2);
        let blank = config.blank_token_id();
        let layout = TdtJointLayout::from(&config);
        let mut predictor = predictor();
        let mut joint = joint(
            layout,
            vec![vec![(blank, 0), (blank, 0)], vec![(blank, 0), (blank, 0)]],
        );

        let hypotheses = TdtGreedySearch::new(config)
            .decode_many(&(), &[2, 1], &mut predictor, &mut joint)
            .unwrap();

        assert!(
            hypotheses
                .iter()
                .all(|hypothesis| hypothesis.tokens().is_empty())
        );
        assert_eq!(predictor.calls, [[blank, blank]]);
        assert_eq!(joint.time_indices, [[0, 0], [1, 0]]);
    }

    #[test]
    fn forces_time_forward_after_the_nonblank_symbol_limit() {
        let config = TdtDecodingConfig::new(8_193, 8_192, [0, 1, 2, 3, 4], 2);
        let blank = config.blank_token_id();
        let layout = TdtJointLayout::from(&config);
        let mut predictor = predictor();
        let mut joint = joint(
            layout,
            vec![
                vec![(10, 0)],
                vec![(11, 0)],
                vec![(12, 0)],
                vec![(blank, 1)],
            ],
        );

        let hypothesis = TdtGreedySearch::new(config)
            .decode(&(), 2, &mut predictor, &mut joint)
            .unwrap();

        assert_eq!(
            hypothesis
                .tokens()
                .iter()
                .map(|token| (token.token_id(), token.start_frame()))
                .collect::<Vec<_>>(),
            [(10, 0), (11, 0), (12, 1)]
        );
        assert_eq!(predictor.calls, [[blank], [10], [11], [12]]);
        assert_eq!(joint.time_indices, [[0], [0], [1], [1]]);
    }
}
