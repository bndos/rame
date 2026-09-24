use crate::RameResult;
use crate::models::ModelError;
use crate::runtime::{JointNetwork, PredictionNetwork, TransducerEncoding};
use crate::tensor::{D, DType, Tensor, TensorError};

use super::workspace::Workspace;
use crate::runtime::transducer::tdt::{
    TdtDecodingConfig, TdtHypothesis, TdtJointOutput, TdtSearch,
};

#[derive(Debug)]
pub struct StandardTdtGreedyExecutor {
    config: TdtDecodingConfig,
    workspace: Option<Workspace>,
}

impl StandardTdtGreedyExecutor {
    pub fn new(config: TdtDecodingConfig) -> Self {
        Self {
            config,
            workspace: None,
        }
    }

    pub fn config(&self) -> &TdtDecodingConfig {
        &self.config
    }
}

fn prepare_workspace<'a>(
    workspace: &'a mut Option<Workspace>,
    device: &crate::tensor::Device,
    encoded_lengths: &Tensor,
    max_encoded_frames: usize,
    config: &TdtDecodingConfig,
) -> RameResult<&'a mut Workspace> {
    let reusable = workspace
        .as_ref()
        .map(|workspace| {
            workspace.is_compatible(
                device,
                encoded_lengths,
                max_encoded_frames,
                config.max_symbols_per_step(),
            )
        })
        .transpose()?
        .unwrap_or(false);
    if !reusable {
        *workspace = Some(Workspace::new(
            device,
            encoded_lengths,
            max_encoded_frames,
            config,
        )?);
    } else if let Some(workspace) = workspace.as_mut() {
        workspace.reset(encoded_lengths)?;
    }

    workspace.as_mut().ok_or_else(|| {
        ModelError::UnsupportedFeature {
            feature: "standard TDT greedy workspace",
            reason: "workspace initialization failed",
        }
        .into()
    })
}

impl<P, J> TdtSearch<P, J> for StandardTdtGreedyExecutor
where
    P: PredictionNetwork,
    J: JointNetwork<Output = TdtJointOutput>,
{
    fn decode_many(
        &mut self,
        encoding: TransducerEncoding,
        predictor: &mut P,
        joint: &mut J,
    ) -> RameResult<Vec<TdtHypothesis>> {
        let (encoded, encoded_lengths) = encoding.into_parts();
        validate_encoding_devices(&encoded, &encoded_lengths)?;
        let batch_size = validate_encoded_lengths(&encoded_lengths)?;
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let config = &self.config;
        let blank_token_id = config.blank_token_id();
        let max_encoded_frames = encoded_length_limit(&encoded_lengths)?;
        let workspace = prepare_workspace(
            &mut self.workspace,
            encoded.device(),
            &encoded_lengths,
            max_encoded_frames,
            config,
        )?;
        let active = workspace
            .time_indices
            .lt(&workspace.encoded_lengths)
            .map_err(TensorError::from)?;
        if !any(&active)? {
            return workspace.hypotheses();
        }
        let initial_state = predictor.initial_state(batch_size)?;
        let (mut predicted, mut state) = predictor.predict(&workspace.labels, &initial_state)?;

        while any(&workspace
            .time_indices
            .lt(&workspace.encoded_lengths)
            .map_err(TensorError::from)?)?
        {
            let active = workspace
                .time_indices
                .lt(&workspace.encoded_lengths)
                .map_err(TensorError::from)?;
            let mut seeking = active.clone();
            workspace.labels = workspace.blank_labels().clone();
            workspace.label_times = workspace.zeros().clone();
            workspace.label_durations = workspace.zeros().clone();

            while any(&seeking)? {
                let last_indices = (&workspace.encoded_lengths - workspace.ones())
                    .and_then(|indices| indices.maximum(workspace.zeros()))
                    .map_err(TensorError::from)?;
                let safe_time_indices = workspace
                    .time_indices
                    .minimum(&last_indices)
                    .map_err(TensorError::from)?;
                let output = joint.joint(&encoded, &safe_time_indices, &predicted)?;
                let token_ids =
                    decision_indices(output.token_logits(), "TDT token logits", batch_size)?;
                let duration_indices =
                    decision_indices(output.duration_logits(), "TDT duration logits", batch_size)?;
                let selected_durations = workspace
                    .durations()
                    .index_select(
                        &duration_indices
                            .to_dtype(DType::U32)
                            .map_err(TensorError::from)?,
                        0,
                    )
                    .map_err(TensorError::from)?;
                let blank = token_ids
                    .eq(i64::from(blank_token_id))
                    .map_err(TensorError::from)?;
                let emitted = mask_and(&seeking, &blank.eq(0u8).map_err(TensorError::from)?)?;

                workspace.labels = emitted
                    .where_cond(&token_ids, &workspace.labels)
                    .map_err(TensorError::from)?;
                workspace.label_times = emitted
                    .where_cond(&workspace.time_indices, &workspace.label_times)
                    .map_err(TensorError::from)?;
                workspace.label_durations = emitted
                    .where_cond(&selected_durations, &workspace.label_durations)
                    .map_err(TensorError::from)?;

                let blank_durations = selected_durations
                    .maximum(workspace.ones())
                    .map_err(TensorError::from)?;
                let advances = blank
                    .where_cond(&blank_durations, &selected_durations)
                    .map_err(TensorError::from)?;
                let advanced = (&workspace.time_indices + &advances).map_err(TensorError::from)?;
                workspace.time_indices = seeking
                    .where_cond(&advanced, &workspace.time_indices)
                    .map_err(TensorError::from)?;
                seeking = mask_and(&seeking, &blank)?;
                seeking = mask_and(
                    &seeking,
                    &workspace
                        .time_indices
                        .lt(&workspace.encoded_lengths)
                        .map_err(TensorError::from)?,
                )?;
            }

            let found = mask_and(
                &active,
                &workspace
                    .labels
                    .ne(i64::from(blank_token_id))
                    .map_err(TensorError::from)?,
            )?;
            if !any(&found)? {
                continue;
            }

            workspace.record(&found)?;
            let same_time = workspace
                .label_times
                .eq(&workspace.last_symbol_times)
                .map_err(TensorError::from)?;
            let next_symbols =
                (&workspace.symbols_at_time + workspace.ones()).map_err(TensorError::from)?;
            let symbol_counts = same_time
                .where_cond(&next_symbols, workspace.ones())
                .map_err(TensorError::from)?;
            workspace.symbols_at_time = found
                .where_cond(&symbol_counts, &workspace.symbols_at_time)
                .map_err(TensorError::from)?;
            workspace.last_symbol_times = found
                .where_cond(&workspace.label_times, &workspace.last_symbol_times)
                .map_err(TensorError::from)?;

            let at_same_frame = workspace
                .time_indices
                .eq(&workspace.label_times)
                .map_err(TensorError::from)?;
            let at_limit = workspace
                .symbols_at_time
                .ge(config.max_symbols_per_step() as i64)
                .map_err(TensorError::from)?;
            let force_advance = mask_and(&mask_and(&found, &at_same_frame)?, &at_limit)?;
            workspace.time_indices = (&workspace.time_indices
                + &force_advance
                    .to_dtype(DType::I64)
                    .map_err(TensorError::from)?)
                .map_err(TensorError::from)?;

            let (candidate_output, candidate_state) =
                predictor.predict(&workspace.labels, &state)?;
            predictor.replace_prediction(
                &mut predicted,
                &mut state,
                candidate_output,
                candidate_state,
                &found,
            )?;
        }

        workspace.hypotheses()
    }
}

fn encoded_length_limit(encoded_lengths: &Tensor) -> RameResult<usize> {
    let encoded_lengths = encoded_lengths
        .to_dtype(DType::I64)
        .map_err(TensorError::from)?;
    let minimum = encoded_lengths
        .min_all()
        .and_then(|minimum| minimum.to_scalar::<i64>())
        .map_err(TensorError::from)?;
    if minimum < 0 {
        return Err(ModelError::UnsupportedFeature {
            feature: "TDT encoded lengths",
            reason: "encoded lengths must be non-negative",
        }
        .into());
    }
    encoded_lengths
        .max_all()
        .and_then(|maximum| maximum.to_scalar::<i64>())
        .map(|maximum| maximum as usize)
        .map_err(TensorError::from)
        .map_err(Into::into)
}

fn validate_encoded_lengths(encoded_lengths: &Tensor) -> RameResult<usize> {
    if encoded_lengths.rank() != 1 {
        return Err(ModelError::InvalidTensorShape {
            name: "TDT encoded lengths".into(),
            expected: "rank 1 tensor".into(),
            actual: encoded_lengths.dims().to_vec(),
        }
        .into());
    }
    Ok(encoded_lengths.elem_count())
}

fn validate_encoding_devices(encoded: &Tensor, encoded_lengths: &Tensor) -> RameResult<()> {
    if !encoded.device().same_device(encoded_lengths.device()) {
        return Err(ModelError::UnsupportedFeature {
            feature: "TDT encoding",
            reason: "encoded activations and lengths must be on the same device",
        }
        .into());
    }
    Ok(())
}

fn decision_indices(logits: &Tensor, name: &'static str, batch_size: usize) -> RameResult<Tensor> {
    let indices = logits
        .argmax(D::Minus1)
        .and_then(|indices| indices.flatten_all())
        .and_then(|indices| indices.to_dtype(DType::I64))
        .map_err(TensorError::from)?;
    if indices.dims() != [batch_size] {
        return Err(ModelError::InvalidTensorShape {
            name: name.into(),
            expected: format!("{batch_size} logit vectors"),
            actual: logits.dims().to_vec(),
        }
        .into());
    }
    Ok(indices)
}

fn mask_and(left: &Tensor, right: &Tensor) -> RameResult<Tensor> {
    (left * right)
        .map_err(TensorError::from)
        .map_err(Into::into)
}

fn any(mask: &Tensor) -> RameResult<bool> {
    mask.to_dtype(DType::U32)
        .and_then(|mask| mask.sum_all())
        .and_then(|sum| sum.to_scalar::<u32>())
        .map(|sum| sum != 0)
        .map_err(TensorError::from)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crate::runtime::transducer::tdt::{
        StandardTdtGreedyExecutor, TdtDecodingConfig, TdtJointLayout, TdtJointOutput, TdtSearch,
    };
    use crate::runtime::{JointNetwork, PredictionNetwork, TransducerEncoding};
    use crate::tensor::{DType, Device, Tensor, TensorError};
    use crate::{RameResult, tokenization::TokenId};

    struct Predictor {
        calls: Vec<Vec<TokenId>>,
    }

    impl PredictionNetwork for Predictor {
        type State = Tensor;

        fn initial_state(&mut self, batch_size: usize) -> RameResult<Self::State> {
            Tensor::zeros(batch_size, DType::I64, &Device::Cpu)
                .map_err(TensorError::from)
                .map_err(Into::into)
        }

        fn predict(
            &mut self,
            tokens: &Tensor,
            state: &Self::State,
        ) -> RameResult<(Tensor, Self::State)> {
            self.calls.push(
                tokens
                    .to_vec1::<i64>()
                    .map_err(TensorError::from)?
                    .into_iter()
                    .map(|token| token as TokenId)
                    .collect(),
            );
            let next_state = (state
                + &Tensor::ones(state.shape(), DType::I64, state.device())
                    .map_err(TensorError::from)?)
                .map_err(TensorError::from)?;
            Ok((tokens.clone(), next_state))
        }

        fn replace_prediction(
            &mut self,
            output: &mut Tensor,
            state: &mut Self::State,
            candidate_output: Tensor,
            candidate_state: Self::State,
            replace_mask: &Tensor,
        ) -> RameResult<()> {
            *output = replace_mask
                .where_cond(&candidate_output, output)
                .map_err(TensorError::from)?;
            *state = replace_mask
                .where_cond(&candidate_state, state)
                .map_err(TensorError::from)?;
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
        type Output = TdtJointOutput;

        fn joint(
            &mut self,
            _encoded: &Tensor,
            time_indices: &Tensor,
            predicted: &Tensor,
        ) -> RameResult<Self::Output> {
            self.time_indices.push(
                time_indices
                    .to_vec1::<i64>()
                    .map_err(TensorError::from)?
                    .into_iter()
                    .map(|time| time as usize)
                    .collect(),
            );
            self.predictions.push(
                predicted
                    .to_vec1::<i64>()
                    .map_err(TensorError::from)?
                    .into_iter()
                    .map(|token| token as TokenId)
                    .collect(),
            );

            let steps = self.steps.pop_front().expect("scripted joint step");
            let mut logits = vec![0.0f32; steps.len() * self.layout.joint_logit_count()];
            for (batch_index, (token_id, duration_index)) in steps.into_iter().enumerate() {
                let offset = batch_index * self.layout.joint_logit_count();
                logits[offset + token_id as usize] = 2.0;
                logits[offset + self.layout.num_token_classes() + duration_index] = 4.0;
            }
            TdtJointOutput::from_combined(
                Tensor::from_vec(
                    logits,
                    (time_indices.elem_count(), self.layout.joint_logit_count()),
                    &Device::Cpu,
                )
                .map_err(TensorError::from)?,
                self.layout,
            )
        }
    }

    fn config(max_symbols_per_step: usize) -> TdtDecodingConfig {
        TdtDecodingConfig::new(4, 3, [0, 1, 2, 3, 4], max_symbols_per_step)
    }

    fn joint(config: &TdtDecodingConfig, steps: Vec<Vec<(TokenId, usize)>>) -> Joint {
        Joint {
            layout: TdtJointLayout::from(config),
            steps: steps.into(),
            time_indices: Vec::new(),
            predictions: Vec::new(),
        }
    }

    fn encoding(lengths: &[u32]) -> TransducerEncoding {
        TransducerEncoding::new(
            Tensor::zeros((lengths.len(), 1, 1), DType::F32, &Device::Cpu).unwrap(),
            Tensor::new(lengths, &Device::Cpu).unwrap(),
        )
    }

    fn tokens(
        hypothesis: &crate::runtime::transducer::tdt::TdtHypothesis,
    ) -> Vec<(u32, usize, usize)> {
        hypothesis
            .tokens()
            .iter()
            .map(|token| (token.token_id(), token.start_frame(), token.duration()))
            .collect()
    }

    #[test]
    fn decodes_heterogeneous_batches_without_recomputing_predictions_for_blanks() {
        let config = config(10);
        let blank = config.blank_token_id();
        let mut predictor = Predictor { calls: Vec::new() };
        let mut joint = joint(
            &config,
            vec![
                vec![(0, 0), (blank, 1)],
                vec![(blank, 4), (1, 1)],
                vec![(2, 2), (blank, 0)],
                vec![(blank, 1), (blank, 0)],
            ],
        );
        let mut search = StandardTdtGreedyExecutor::new(config);

        let hypotheses = search
            .decode_many(encoding(&[3, 2]), &mut predictor, &mut joint)
            .unwrap();

        assert_eq!(tokens(&hypotheses[0]), [(0, 0, 0), (2, 0, 2)]);
        assert_eq!(tokens(&hypotheses[1]), [(1, 1, 1)]);
        assert_eq!(predictor.calls, [[blank, blank], [0, 1], [2, blank]]);
        assert_eq!(joint.time_indices, [[0, 0], [0, 1], [0, 1], [2, 1]]);
        assert_eq!(
            joint.predictions,
            [
                vec![blank, blank],
                vec![blank, blank],
                vec![0, 1],
                vec![2, 1]
            ]
        );
    }

    #[test]
    fn decodes_one_with_the_same_batched_executor() {
        let config = config(10);
        let blank = config.blank_token_id();
        let mut predictor = Predictor { calls: Vec::new() };
        let mut joint = joint(&config, vec![vec![(0, 1)], vec![(blank, 1)]]);
        let mut search = StandardTdtGreedyExecutor::new(config);

        let hypothesis = search
            .decode(encoding(&[2]), &mut predictor, &mut joint)
            .unwrap();

        assert_eq!(tokens(&hypothesis), [(0, 0, 1)]);
        assert_eq!(predictor.calls, [[blank], [0]]);
        assert_eq!(joint.time_indices, [[0], [1]]);
    }

    #[test]
    fn forces_zero_duration_blanks_to_advance() {
        let config = config(2);
        let blank = config.blank_token_id();
        let mut predictor = Predictor { calls: Vec::new() };
        let mut joint = joint(
            &config,
            vec![vec![(blank, 0), (blank, 0)], vec![(blank, 0), (blank, 0)]],
        );
        let mut search = StandardTdtGreedyExecutor::new(config);

        let hypotheses = search
            .decode_many(encoding(&[2, 1]), &mut predictor, &mut joint)
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
    fn advances_after_the_nonblank_symbol_limit() {
        let config = config(2);
        let blank = config.blank_token_id();
        let mut predictor = Predictor { calls: Vec::new() };
        let mut joint = joint(
            &config,
            vec![vec![(0, 0)], vec![(1, 0)], vec![(2, 0)], vec![(blank, 1)]],
        );
        let mut search = StandardTdtGreedyExecutor::new(config);

        let hypothesis = search
            .decode(encoding(&[2]), &mut predictor, &mut joint)
            .unwrap();

        assert_eq!(tokens(&hypothesis), [(0, 0, 0), (1, 0, 0), (2, 1, 0)]);
        assert_eq!(joint.time_indices, [[0], [0], [1], [1]]);
    }

    #[test]
    fn resets_and_reuses_output_storage_between_decodes() {
        let config = config(2);
        let blank = config.blank_token_id();
        let mut predictor = Predictor { calls: Vec::new() };
        let mut search = StandardTdtGreedyExecutor::new(config.clone());
        let mut first_joint = joint(&config, vec![vec![(0, 1)], vec![(blank, 2)]]);

        let first = search
            .decode(encoding(&[3]), &mut predictor, &mut first_joint)
            .unwrap();
        assert_eq!(tokens(&first), [(0, 0, 1)]);

        let mut second_joint = joint(&config, vec![vec![(2, 1)], vec![(blank, 1)]]);
        let second = search
            .decode(encoding(&[2]), &mut predictor, &mut second_joint)
            .unwrap();

        assert_eq!(tokens(&second), [(2, 0, 1)]);
    }
}
