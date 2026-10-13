use std::path::{Path, PathBuf};

use clap::Parser;
use rame::audio::Audio;
use rame::models::parakeet::mlx;
use rame::runtime::ModelLoader;
use rame::transcription::{TranscriptionInput, TranscriptionModel};

#[derive(Parser)]
struct Args {
    /// Directory containing the Parakeet checkpoint and config.
    model_path: PathBuf,
    /// Mono 16-bit PCM WAV to transcribe.
    wav_path: PathBuf,
    /// Use an explicitly quantized Q8 checkpoint.
    #[arg(long)]
    q8: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let audio = load_pcm16_mono(&args.wav_path)?;
    let artifact = if args.q8 {
        mlx::Artifact::default()
            .weights(mlx::WeightFormat::QuantizedQ8)
            .files(mlx::Files {
                model: "model.int8.safetensors".into(),
                ..mlx::Files::default()
            })
    } else {
        mlx::Artifact::default()
    };
    let mut model = artifact.load(args.model_path.as_path())?;
    let result = model.transcribe(TranscriptionInput::new(audio.as_view()))?;
    for chunk in result.results {
        if let Some(alternative) = chunk.alternatives.first() {
            println!("{}", alternative.text);
        }
    }
    Ok(())
}

fn load_pcm16_mono(path: &Path) -> Result<Audio, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        return Err(format!("{} must be mono 16-bit PCM", path.display()).into());
    }
    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.map(|value| value as f32 / 32768.0))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Audio::from_interleaved_f32(
        spec.sample_rate,
        spec.channels,
        samples,
    )?)
}
