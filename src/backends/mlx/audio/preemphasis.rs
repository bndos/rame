use mlx_rs::Array;
use mlx_rs::ops::concatenate;
use mlx_rs::ops::indexing::IndexOp;

use super::state::{MlxAudioBackend, Stage};
use crate::RameResult;
use crate::preprocess::audio::PreEmphasis;
use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp};

impl PreprocessOp<MlxAudioBackend> for PreEmphasis {
    fn forward<'a>(
        &self,
        mut data: <MlxAudioBackend as PreprocessBackend>::Data<'a>,
    ) -> RameResult<<MlxAudioBackend as PreprocessBackend>::Data<'a>> {
        if !self.coefficient.is_finite() {
            return Err(super::invalid("preemphasis coefficient must be finite"));
        }
        let Stage::Pcm(sources) = data.stage else {
            return Err(super::invalid("preemphasis expects PCM input"));
        };
        let mut signals = Vec::with_capacity(sources.len());
        for source in sources {
            let samples = source.data();
            if samples.len() < 2 || i32::try_from(samples.len()).is_err() {
                return Err(super::invalid(
                    "audio sample count is invalid for MLX preprocessing",
                ));
            }
            let array = Array::from_slice(samples, &[samples.len() as i32]);
            signals.push(
                concatenate(
                    &[
                        array.index(..1),
                        array.index(1..) - array.index(..-1) * self.coefficient,
                    ],
                    0,
                )
                .map_err(super::mlx_error)?,
            );
        }
        data.stage = Stage::Signal(signals);
        Ok(data)
    }
}
