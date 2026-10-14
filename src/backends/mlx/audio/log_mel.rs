use mlx_rs::fft::rfft;
use mlx_rs::ops::indexing::{Ellipsis, IndexOp, IntoStrideBy};
use mlx_rs::ops::{as_strided, concatenate};

use super::state::{MlxAudioBackend, Stage};
use crate::RameResult;
use crate::preprocess::audio::LogMelSpectrogram;
use crate::preprocess::pipeline::{PreprocessBackend, PreprocessOp};

impl PreprocessOp<MlxAudioBackend> for LogMelSpectrogram {
    fn forward<'a>(
        &self,
        mut data: <MlxAudioBackend as PreprocessBackend>::Data<'a>,
    ) -> RameResult<<MlxAudioBackend as PreprocessBackend>::Data<'a>> {
        if *self != data.spectrogram {
            return Err(super::invalid(
                "log-mel operation does not match the compiled MLX backend",
            ));
        }
        let Stage::Signal(signals) = data.stage else {
            return Err(super::invalid(
                "log-mel spectrogram expects preemphasized signal",
            ));
        };
        let mut features = Vec::with_capacity(signals.len());
        for signal in signals {
            let padding = (self.fft_size / 2) as i32;
            if signal.size() <= padding as usize {
                return Err(super::invalid("audio is too short for reflect padding"));
            }
            let prefix = signal.index(1..padding + 1).index((..).stride_by(-1));
            let suffix = signal.index(-(padding + 1)..-1).index((..).stride_by(-1));
            let reflected = concatenate(&[prefix, signal, suffix], 0).map_err(super::mlx_error)?;
            let frame_count =
                (reflected.size() - self.window_size + self.hop_length) / self.hop_length;
            let frames = as_strided(
                reflected,
                &[frame_count as i32, self.fft_size as i32],
                &[self.hop_length as i64, 1],
                None,
            )
            .map_err(super::mlx_error)?;
            let spectrum = rfft(frames * &data.assets.window, self.fft_size as i32, -1)
                .map_err(super::mlx_error)?;
            let components = spectrum
                .view::<f32>()
                .map_err(super::mlx_error)?
                .abs()
                .map_err(super::mlx_error)?;
            let magnitude = components.index((Ellipsis, (..).stride_by(2)))
                + components.index((Ellipsis, (1..).stride_by(2)));
            let power = magnitude.square().map_err(super::mlx_error)?;
            let mel = data
                .assets
                .filterbank
                .matmul(power.transpose_axes(&[1, 0]).map_err(super::mlx_error)?)
                .map_err(super::mlx_error)?;
            features.push((mel + self.log_guard).log().map_err(super::mlx_error)?);
        }
        data.stage = Stage::LogMel(features);
        Ok(data)
    }
}
