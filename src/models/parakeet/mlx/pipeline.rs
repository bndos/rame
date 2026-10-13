use std::path::Path;
use std::time::Duration;

use mlx_rs::{Array, Device, Stream, with_device, with_stream};

use super::artifact::{Preprocess, WeightFormat};
use super::decoder::{HIDDEN, TensorJoint, TensorPredictor};
use super::network::Encoder;
use crate::models::ModelError;
use crate::models::parakeet::ParakeetTdtConfig;
use crate::preprocess::audio::{
    self, AudioPipeline, LogMelSpectrogram, NormalizeFeatures, PreEmphasis,
};
use crate::runtime::transducer::tdt::{
    StandardTdtGreedyExecutor, TdtHypothesis, TdtJointLayout, TdtSearch,
};
use crate::runtime::{
    ProcessedBatch, Processor, TransducerDecoding, TransducerEncoder, TransducerEncoding,
    TransducerModelRunner,
};
use crate::tensor::{DType, Device as TensorDevice, Tensor, TensorError};
use crate::tokenization::{Decoder, SentencePieceDecoder};
use crate::transcription::{
    AudioChannelMode, TimeSpan, TimestampGranularity, TranscriptionAlternative, TranscriptionChunk,
    TranscriptionError, TranscriptionInput, TranscriptionOptions, TranscriptionResult,
};
use crate::{RameError, RameResult};

pub type ParakeetMlxModel =
    TransducerModelRunner<MlxPreprocessor, HostEncoder, TensorPredictor, TensorJoint, CpuDecoding>;

pub struct MlxPreprocessor {
    pipeline: AudioPipeline,
    sample_rate: u32,
    stream: Stream,
}

pub struct MlxEncoder {
    network: Encoder,
    stream: Stream,
}

pub struct HostEncoder(MlxEncoder);

pub struct Decoding<S> {
    search: S,
    decoder: SentencePieceDecoder,
}

pub struct CpuDecoding(Decoding<StandardTdtGreedyExecutor>);

pub struct Context {
    duration: Duration,
    timestamps: TimestampGranularity,
}

pub(super) fn load(
    preprocess: Preprocess,
    weights_format: WeightFormat,
    mut config: ParakeetTdtConfig,
    decoder: SentencePieceDecoder,
    model_path: &Path,
) -> RameResult<ParakeetMlxModel> {
    let log_mel = LogMelSpectrogram::new(
        preprocess.sample_rate,
        preprocess.feature_size,
        preprocess.n_fft,
        preprocess.win_length,
        preprocess.hop_length,
    );
    let pipeline = audio::pipeline(log_mel)?
        .add_op(PreEmphasis::new(preprocess.preemphasis))
        .add_op(log_mel)
        .add_op(NormalizeFeatures::new(DType::BF16))
        .compile();
    config.feature_extractor = preprocess;
    let (mut weights, stream) = load_parts(model_path)?;
    let predictor = with_stream(&stream, || TensorPredictor::load(&mut weights))
        .map_err(|error| mlx_error("construct Parakeet predictor", error))?;
    let joint = with_stream(&stream, || {
        TensorJoint::load(
            &mut weights,
            TdtJointLayout::new(config.num_token_classes, config.durations.len()),
        )
    })
    .map_err(|error| mlx_error("construct Parakeet joint", error))?;
    let network = with_stream(&stream, || Encoder::load(&mut weights, weights_format))
        .map_err(|error| mlx_error("construct Parakeet encoder", error))?;
    Ok(TransducerModelRunner::new(
        MlxPreprocessor {
            pipeline,
            sample_rate: preprocess.sample_rate,
            stream: stream.clone(),
        },
        HostEncoder(MlxEncoder { network, stream }),
        predictor,
        joint,
        CpuDecoding(Decoding {
            search: StandardTdtGreedyExecutor::new(config.into()),
            decoder,
        }),
    ))
}

fn load_parts(model_path: &Path) -> RameResult<(std::collections::HashMap<String, Array>, Stream)> {
    let weights = with_device(Device::cpu(), || Array::load_safetensors(model_path))
        .map_err(|error| mlx_error("load Parakeet weights", error))?;
    let stream = Stream::try_default_on_device(&Device::gpu())
        .map_err(|error| mlx_error("open default Parakeet GPU stream", error))?;
    Ok((weights, stream))
}

impl Processor for MlxPreprocessor {
    type Source<'a> = TranscriptionInput<'a>;
    type Context = Context;
    type Inputs = Vec<Array>;

    fn process_many<'a>(
        &self,
        sources: &'a [Self::Source<'a>],
    ) -> RameResult<ProcessedBatch<Self::Context, Self::Inputs>> {
        let mut views = Vec::with_capacity(sources.len());
        let mut contexts = Vec::with_capacity(sources.len());
        for source in sources {
            validate_input(source, self.sample_rate)?;
            views.push(source.audio);
            contexts.push(Context {
                duration: source.audio.duration(),
                timestamps: source.options.timestamps,
            });
        }
        let inputs = with_stream(&self.stream, || self.pipeline.process_many(&views))?;
        Ok(ProcessedBatch {
            len: sources.len(),
            inputs,
            contexts,
        })
    }
}

impl MlxEncoder {
    fn encode_outputs(&mut self, inputs: Vec<Array>) -> RameResult<(Vec<Array>, Vec<usize>)> {
        let mut encoded = Vec::with_capacity(inputs.len());
        let mut lengths = Vec::with_capacity(inputs.len());
        for features in inputs {
            let output = with_stream(&self.stream, || self.network.forward(&features))
                .map_err(|error| mlx_error("encode Parakeet features", error))?;
            lengths.push(output.dim(1) as usize);
            encoded.push(output);
        }
        Ok((encoded, lengths))
    }
}

impl TransducerEncoder for HostEncoder {
    type Inputs = Vec<Array>;
    type Encoded = Tensor;
    type Lengths = Tensor;

    fn encode(
        &mut self,
        inputs: Self::Inputs,
    ) -> RameResult<TransducerEncoding<Self::Encoded, Self::Lengths>> {
        let (encoded, lengths) = self.0.encode_outputs(inputs)?;
        let encoded = encoded
            .into_iter()
            .map(|output| {
                output.to_vec_cast::<f32>().map_err(|error| {
                    ModelError::InvalidInput(format!("read Parakeet encoded frames: {error}"))
                        .into()
                })
            })
            .collect::<RameResult<Vec<_>>>()?;
        let batch = encoded.len();
        let max_frames = lengths.iter().copied().max().unwrap_or(0);
        let width = HIDDEN;
        let capacity = batch
            .checked_mul(max_frames)
            .and_then(|count| count.checked_mul(width))
            .ok_or_else(|| {
                ModelError::InvalidInput("Parakeet encoded batch is too large".into())
            })?;
        let mut values = if batch == 1 {
            Vec::new()
        } else {
            Vec::with_capacity(capacity)
        };
        for (row, frames) in encoded.into_iter().enumerate() {
            let expected = lengths[row].checked_mul(width).ok_or_else(|| {
                ModelError::InvalidInput("Parakeet encoded frame count is too large".into())
            })?;
            if frames.len() != expected {
                return Err(ModelError::InvalidTensorShape {
                    name: "Parakeet encoded frames".into(),
                    expected: format!("{expected} values"),
                    actual: vec![frames.len()],
                }
                .into());
            }
            if batch == 1 {
                values = frames;
            } else {
                values.extend(frames);
                values.resize(values.len() + (max_frames - lengths[row]) * width, 0.0);
            }
        }
        let encoded = Tensor::from_vec(values, (batch, max_frames, width), &TensorDevice::Cpu)
            .map_err(TensorError::from)?;
        let lengths = Tensor::from_vec(
            lengths.into_iter().map(|length| length as i64).collect(),
            batch,
            &TensorDevice::Cpu,
        )
        .map_err(TensorError::from)?;
        Ok(TransducerEncoding::new(encoded, lengths))
    }
}

impl<E, P, J, S> TransducerDecoding<E, P, J> for Decoding<S>
where
    E: TransducerEncoder,
    P: crate::runtime::PredictionNetwork,
    J: crate::runtime::JointNetwork,
    S: TdtSearch<P, J, Encoding = TransducerEncoding<E::Encoded, E::Lengths>>,
{
    type Context = Context;
    type Output = TranscriptionResult;

    fn decode_many(
        &mut self,
        encoding: TransducerEncoding<E::Encoded, E::Lengths>,
        contexts: &[Self::Context],
        predictor: &mut P,
        joint: &mut J,
    ) -> RameResult<Vec<Self::Output>> {
        let hypotheses = self.search.decode_many(encoding, predictor, joint)?;
        self.finish_many(&hypotheses, contexts)
    }
}

impl TransducerDecoding<HostEncoder, TensorPredictor, TensorJoint> for CpuDecoding {
    type Context = Context;
    type Output = TranscriptionResult;

    fn decode_many(
        &mut self,
        encoding: TransducerEncoding,
        contexts: &[Self::Context],
        predictor: &mut TensorPredictor,
        joint: &mut TensorJoint,
    ) -> RameResult<Vec<Self::Output>> {
        let (encoded, lengths) = encoding.into_parts();
        let batch = lengths.dim(0).map_err(TensorError::from)?;
        let encoded_batch = encoded.dim(0).map_err(TensorError::from)?;
        if encoded_batch != batch {
            return Err(RameError::InvalidBatchLength {
                stage: "Parakeet encoded rows",
                expected: batch,
                actual: encoded_batch,
            });
        }
        let mut hypotheses = Vec::with_capacity(batch);
        for row in 0..batch {
            let encoded_row = encoded.narrow(0, row, 1).map_err(TensorError::from)?;
            let length_row = lengths.narrow(0, row, 1).map_err(TensorError::from)?;
            hypotheses.push(self.0.search.decode(
                TransducerEncoding::new(encoded_row, length_row),
                predictor,
                joint,
            )?);
        }
        self.0.finish_many(&hypotheses, contexts)
    }
}

impl<S> Decoding<S> {
    fn finish_many(
        &self,
        hypotheses: &[TdtHypothesis],
        contexts: &[Context],
    ) -> RameResult<Vec<TranscriptionResult>> {
        if hypotheses.len() != contexts.len() {
            return Err(RameError::InvalidBatchLength {
                stage: "Parakeet TDT hypotheses",
                expected: contexts.len(),
                actual: hypotheses.len(),
            });
        }
        hypotheses
            .iter()
            .zip(contexts)
            .map(|(hypothesis, context)| self.finish_result(hypothesis, context))
            .collect()
    }

    fn finish_result(
        &self,
        hypothesis: &TdtHypothesis,
        context: &Context,
    ) -> RameResult<TranscriptionResult> {
        let token_ids = hypothesis
            .tokens()
            .iter()
            .map(|alignment| alignment.token_id())
            .collect::<Vec<_>>();
        let text = self.decoder.decode(&token_ids, false)?;
        let time = if context.timestamps == TimestampGranularity::Segment {
            Some(TimeSpan::new(Duration::ZERO, context.duration)?)
        } else {
            None
        };
        Ok(TranscriptionResult {
            results: vec![TranscriptionChunk {
                time,
                alternatives: vec![TranscriptionAlternative {
                    text,
                    confidence: None,
                    words: Vec::new(),
                }],
                language: Some("en".to_string()),
                channel: Some(0),
            }],
            audio_duration: context.duration,
        })
    }
}

fn validate_input(input: &TranscriptionInput<'_>, sample_rate: u32) -> RameResult<()> {
    if input.audio.sample_rate() != sample_rate || input.audio.channels() != 1 {
        return Err(ModelError::InvalidInput(format!(
            "Parakeet MLX expects {sample_rate} Hz mono audio, got {} Hz with {} channels",
            input.audio.sample_rate(),
            input.audio.channels()
        ))
        .into());
    }
    validate_options(&input.options)
}

fn validate_options(options: &TranscriptionOptions) -> RameResult<()> {
    if !options.languages.is_empty()
        && !options.languages.iter().all(|language| {
            language.eq_ignore_ascii_case("en") || language.eq_ignore_ascii_case("en-us")
        })
    {
        return Err(unsupported("languages"));
    }
    if matches!(
        options.timestamps,
        TimestampGranularity::Word | TimestampGranularity::All
    ) {
        return Err(unsupported("timestamps"));
    }
    if options.word_confidence {
        return Err(unsupported("word_confidence"));
    }
    if options.max_alternatives.get() != 1 {
        return Err(unsupported("max_alternatives"));
    }
    if options.diarization.is_some() {
        return Err(unsupported("diarization"));
    }
    if matches!(options.channel_mode, AudioChannelMode::Separate) {
        return Err(unsupported("channel_mode"));
    }
    Ok(())
}

fn unsupported(option: &'static str) -> RameError {
    TranscriptionError::UnsupportedOption { option }.into()
}

fn mlx_error(stage: &str, error: impl std::fmt::Display) -> RameError {
    ModelError::InvalidArtifact(format!("failed to {stage}: {error}")).into()
}
