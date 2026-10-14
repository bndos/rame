use mlx_rs::ops::indexing::{IndexOp, argmax_axis};
use mlx_rs::ops::{maximum, select, stack};
use mlx_rs::{Array, Stream, with_stream};

use crate::RameResult;
use crate::runtime::RuntimeError;
use crate::runtime::transducer::tdt::{
    TdtDecodingConfig, TdtHypothesis, TdtJointLayout, TdtSearch, TdtTokenAlignment,
};
use crate::runtime::{JointNetwork, MaskedPredictionNetwork, TransducerEncoding};

pub struct MlxTdtGreedyExecutor {
    config: TdtDecodingConfig,
    stream: Stream,
}

impl MlxTdtGreedyExecutor {
    pub fn new(config: TdtDecodingConfig, stream: Stream) -> Self {
        Self { config, stream }
    }

    fn decode_on_stream<P, J>(
        &mut self,
        encoding: TransducerEncoding<Array, Vec<usize>>,
        predictor: &mut P,
        joint: &mut J,
    ) -> RameResult<Vec<TdtHypothesis>>
    where
        P: MaskedPredictionNetwork<Tokens = Array, Output = Array, Mask = Array>,
        J: JointNetwork<Encoded = Array, Indices = Array, Predicted = Array, Output = Array>,
    {
        let (encoded, lengths) = encoding.into_parts();
        let [batch, frames, _] = encoded.shape() else {
            return Err(invalid_shape(
                "TDT encoded frames",
                "[batch, frames, width]",
                encoded.shape(),
            ));
        };
        if lengths.len() != *batch as usize {
            return Err(crate::RameError::InvalidBatchLength {
                stage: "TDT encoded lengths",
                expected: *batch as usize,
                actual: lengths.len(),
            });
        }
        if self.config.num_token_classes() == 0
            || self.config.durations().is_empty()
            || self.config.max_symbols_per_step() == 0
        {
            return Err(RuntimeError::InvalidInput(
                "TDT class, duration, and symbol limits must be non-zero".into(),
            )
            .into());
        }
        if i32::try_from(self.config.num_token_classes()).is_err()
            || self
                .config
                .num_token_classes()
                .checked_add(self.config.durations().len())
                .and_then(|count| i32::try_from(count).ok())
                .is_none()
            || i32::try_from(self.config.max_symbols_per_step()).is_err()
            || self.config.blank_token_id() as usize >= self.config.num_token_classes()
        {
            return Err(RuntimeError::InvalidInput(
                "TDT class, blank, or symbol limit is invalid for MLX".into(),
            )
            .into());
        }
        if lengths.iter().any(|&length| length > *frames as usize) {
            return Err(RuntimeError::InvalidInput(
                "TDT encoded length exceeds frame count".into(),
            )
            .into());
        }
        if *batch == 0 {
            return Ok(Vec::new());
        }
        let lengths = lengths
            .into_iter()
            .map(|value| {
                i32::try_from(value).map_err(|_| {
                    RuntimeError::InvalidInput("TDT encoded length exceeds i32".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let blank_id = i32::try_from(self.config.blank_token_id())
            .map_err(|_| RuntimeError::InvalidInput("TDT blank token exceeds i32".into()))?;
        let durations = self
            .config
            .durations()
            .iter()
            .map(|&value| {
                i32::try_from(value)
                    .map_err(|_| RuntimeError::InvalidInput("TDT duration exceeds i32".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let max_duration = durations.iter().copied().max().unwrap_or(0).max(1);
        if lengths.iter().copied().max().unwrap_or(0) > i32::MAX - max_duration {
            return Err(
                RuntimeError::InvalidInput("TDT time index may overflow i32".into()).into(),
            );
        }
        let lengths = Array::from_slice(&lengths, &[*batch]);
        let durations = Array::from_slice(&durations, &[durations.len() as i32]);
        let zeros = Array::zeros::<i32>(&[*batch]).map_err(mlx_error)?;
        let ones = Array::ones::<i32>(&[*batch]).map_err(mlx_error)?;
        let blank = &zeros + blank_id;
        let mut time = zeros.clone();
        let mut symbols_at_time = zeros.clone();
        let mut last_symbol_times = &zeros - 1i32;
        let mut records = Vec::new();
        if !any(&time.lt(&lengths).map_err(mlx_error)?)? {
            return Ok(vec![TdtHypothesis::new(Vec::new()); *batch as usize]);
        }
        let initial_state = predictor.initial_state(*batch as usize)?;
        let (mut predicted, mut state) = predictor.predict(&blank, &initial_state)?;
        let layout = TdtJointLayout::from(&self.config);

        while any(&time.lt(&lengths).map_err(mlx_error)?)? {
            let active = time.lt(&lengths).map_err(mlx_error)?;
            let mut seeking = active.clone();
            let mut labels = blank.clone();
            let mut label_times = zeros.clone();
            let mut label_durations = zeros.clone();
            while any(&seeking)? {
                let last = maximum(&(&lengths - 1i32), &zeros).map_err(mlx_error)?;
                let safe_time = mlx_rs::ops::minimum(&time, &last).map_err(mlx_error)?;
                let logits = joint.joint(&encoded, &safe_time, &predicted)?;
                if logits.shape() != [*batch, layout.joint_logit_count() as i32] {
                    return Err(invalid_shape(
                        "TDT joint logits",
                        "[batch, token classes + durations]",
                        logits.shape(),
                    ));
                }
                let tokens = argmax_axis(
                    logits.index((.., ..layout.num_token_classes() as i32)),
                    -1,
                    None,
                )
                .map_err(mlx_error)?;
                let duration_ids = argmax_axis(
                    logits.index((.., layout.num_token_classes() as i32..)),
                    -1,
                    None,
                )
                .map_err(mlx_error)?;
                let duration = durations.take(&duration_ids).map_err(mlx_error)?;
                let is_blank = tokens.eq(&blank).map_err(mlx_error)?;
                let emitted = seeking
                    .logical_and(&is_blank.logical_not().map_err(mlx_error)?)
                    .map_err(mlx_error)?;
                labels = select(&emitted, &tokens, &labels).map_err(mlx_error)?;
                label_times = select(&emitted, &time, &label_times).map_err(mlx_error)?;
                label_durations =
                    select(&emitted, &duration, &label_durations).map_err(mlx_error)?;
                let blank_duration = maximum(&duration, &ones).map_err(mlx_error)?;
                let advance = select(&is_blank, &blank_duration, &duration).map_err(mlx_error)?;
                time = select(&seeking, &(&time + &advance), &time).map_err(mlx_error)?;
                seeking = seeking
                    .logical_and(&is_blank)
                    .map_err(mlx_error)?
                    .logical_and(&time.lt(&lengths).map_err(mlx_error)?)
                    .map_err(mlx_error)?;
            }
            let found = active
                .logical_and(&labels.ne(&blank).map_err(mlx_error)?)
                .map_err(mlx_error)?;
            if !any(&found)? {
                continue;
            }
            records.push(
                stack(
                    &[
                        &labels,
                        &label_times,
                        &label_durations,
                        &found.as_dtype(mlx_rs::Dtype::Int32).map_err(mlx_error)?,
                    ],
                    -1,
                )
                .map_err(mlx_error)?,
            );
            let same_time = label_times.eq(&last_symbol_times).map_err(mlx_error)?;
            let count = select(&same_time, &(&symbols_at_time + 1i32), &ones).map_err(mlx_error)?;
            symbols_at_time = select(&found, &count, &symbols_at_time).map_err(mlx_error)?;
            last_symbol_times =
                select(&found, &label_times, &last_symbol_times).map_err(mlx_error)?;
            let force_advance = found
                .logical_and(&time.eq(&label_times).map_err(mlx_error)?)
                .map_err(mlx_error)?
                .logical_and(
                    &symbols_at_time
                        .ge(&(&zeros + self.config.max_symbols_per_step() as i32))
                        .map_err(mlx_error)?,
                )
                .map_err(mlx_error)?;
            time = &time
                + force_advance
                    .as_dtype(mlx_rs::Dtype::Int32)
                    .map_err(mlx_error)?;
            let (candidate_output, candidate_state) = predictor.predict(&labels, &state)?;
            predictor.replace_prediction(
                &mut predicted,
                &mut state,
                candidate_output,
                candidate_state,
                &found,
            )?;
        }

        let mut hypotheses = vec![TdtHypothesis::new(Vec::new()); *batch as usize];
        if records.is_empty() {
            return Ok(hypotheses);
        }
        let output = stack(&records, 1).map_err(mlx_error)?;
        let values = output
            .to_vec_cast::<i32>()
            .map_err(|error| RuntimeError::backend("MLX", "read TDT output", error))?;
        for (row, hypothesis) in hypotheses.iter_mut().enumerate() {
            for step in 0..records.len() {
                let offset = (row * records.len() + step) * 4;
                if values[offset + 3] != 0 {
                    hypothesis.push(TdtTokenAlignment::new(
                        values[offset] as u32,
                        values[offset + 1] as usize,
                        values[offset + 2] as usize,
                    ));
                }
            }
        }
        Ok(hypotheses)
    }
}

impl<P, J> TdtSearch<P, J> for MlxTdtGreedyExecutor
where
    P: MaskedPredictionNetwork<Tokens = Array, Output = Array, Mask = Array>,
    J: JointNetwork<Encoded = Array, Indices = Array, Predicted = Array, Output = Array>,
{
    type Encoding = TransducerEncoding<Array, Vec<usize>>;

    fn decode_many(
        &mut self,
        encoding: Self::Encoding,
        predictor: &mut P,
        joint: &mut J,
    ) -> RameResult<Vec<TdtHypothesis>> {
        let stream = self.stream.clone();
        with_stream(&stream, || {
            self.decode_on_stream(encoding, predictor, joint)
        })
    }
}

fn any(values: &Array) -> RameResult<bool> {
    values
        .any(None)
        .map_err(mlx_error)?
        .try_item_exact::<bool>()
        .map_err(|error| RuntimeError::backend("MLX", "check TDT completion", error).into())
}

fn mlx_error(error: mlx_rs::error::Exception) -> crate::RameError {
    RuntimeError::backend("MLX", "TDT search", error).into()
}

fn invalid_shape(name: &str, expected: &str, actual: &[i32]) -> crate::RameError {
    RuntimeError::InvalidTensorShape {
        name: name.into(),
        expected: expected.into(),
        actual: actual.iter().map(|&dimension| dimension as usize).collect(),
    }
    .into()
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use mlx_rs::ops::select;
    use mlx_rs::{Array, Device, Stream};

    use crate::RameResult;
    use crate::runtime::RuntimeError;
    use crate::runtime::transducer::tdt::{TdtDecodingConfig, TdtSearch};
    use crate::runtime::{
        JointNetwork, MaskedPredictionNetwork, PredictionNetwork, TransducerEncoding,
    };

    use super::MlxTdtGreedyExecutor;

    struct Predictor;

    impl PredictionNetwork for Predictor {
        type Tokens = Array;
        type Output = Array;
        type State = Array;

        fn initial_state(&mut self, batch_size: usize) -> RameResult<Array> {
            Array::zeros::<i32>(&[batch_size as i32]).map_err(super::mlx_error)
        }

        fn predict(&mut self, _: &Array, state: &Array) -> RameResult<(Array, Array)> {
            Ok((state.clone(), state + 1i32))
        }
    }

    impl MaskedPredictionNetwork for Predictor {
        type Mask = Array;

        fn replace_prediction(
            &mut self,
            output: &mut Array,
            state: &mut Array,
            candidate_output: Array,
            candidate_state: Array,
            mask: &Array,
        ) -> RameResult<()> {
            *output = select(mask, &candidate_output, &*output).map_err(super::mlx_error)?;
            *state = select(mask, &candidate_state, &*state).map_err(super::mlx_error)?;
            Ok(())
        }
    }

    struct Joint {
        decisions: VecDeque<[(i32, i32); 2]>,
        predictions: Vec<Vec<i32>>,
    }

    impl JointNetwork for Joint {
        type Encoded = Array;
        type Indices = Array;
        type Predicted = Array;
        type Output = Array;

        fn joint(&mut self, _: &Array, _: &Array, predicted: &Array) -> RameResult<Array> {
            self.predictions.push(
                predicted
                    .to_vec_cast::<i32>()
                    .map_err(|error| RuntimeError::InvalidInput(error.to_string()))?,
            );
            let decisions = self.decisions.pop_front().expect("scripted TDT decision");
            let mut logits = [0.0f32; 10];
            for (row, (token, duration)) in decisions.into_iter().enumerate() {
                logits[row * 5 + token as usize] = 2.0;
                logits[row * 5 + 3 + duration as usize] = 3.0;
            }
            Ok(Array::from_slice(&logits, &[2, 5]))
        }
    }

    #[test]
    fn batched_search_masks_finished_rows_and_commits_only_emissions() {
        let stream = Stream::try_default_on_device(&Device::cpu()).unwrap();
        let mut search = MlxTdtGreedyExecutor::new(TdtDecodingConfig::new(3, 2, [0, 1], 2), stream);
        let mut predictor = Predictor;
        let mut joint = Joint {
            decisions: VecDeque::from([
                [(0, 0), (2, 1)],
                [(1, 0), (2, 1)],
                [(2, 1), (2, 1)],
                [(0, 1), (2, 1)],
            ]),
            predictions: Vec::new(),
        };
        let encoded = Array::zeros::<f32>(&[2, 3, 1]).unwrap();
        let hypotheses = search
            .decode_many(
                TransducerEncoding::new(encoded, vec![3, 1]),
                &mut predictor,
                &mut joint,
            )
            .unwrap();
        assert_eq!(hypotheses.len(), 2);
        assert_eq!(
            hypotheses[0]
                .tokens()
                .iter()
                .map(|token| (token.token_id(), token.start_frame(), token.duration()))
                .collect::<Vec<_>>(),
            [(0, 0, 0), (1, 0, 0), (0, 2, 1)]
        );
        assert!(hypotheses[1].tokens().is_empty());
        assert_eq!(
            joint.predictions,
            [vec![0, 0], vec![1, 0], vec![2, 0], vec![2, 0]]
        );
    }

    #[test]
    fn zero_duration_blank_makes_progress() {
        let stream = Stream::try_default_on_device(&Device::cpu()).unwrap();
        let mut search = MlxTdtGreedyExecutor::new(TdtDecodingConfig::new(3, 2, [0, 1], 2), stream);
        let mut predictor = Predictor;
        let mut joint = Joint {
            decisions: VecDeque::from([[(2, 0), (2, 0)], [(2, 0), (2, 0)]]),
            predictions: Vec::new(),
        };
        let encoded = Array::zeros::<f32>(&[2, 2, 1]).unwrap();
        let hypotheses = search
            .decode_many(
                TransducerEncoding::new(encoded, vec![2, 1]),
                &mut predictor,
                &mut joint,
            )
            .unwrap();
        assert!(
            hypotheses
                .iter()
                .all(|hypothesis| hypothesis.tokens().is_empty())
        );
        assert_eq!(joint.predictions.len(), 2);
    }
}
