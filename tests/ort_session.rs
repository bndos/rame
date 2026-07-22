#![cfg(feature = "onnxruntime")]

use std::path::Path;

use ndarray::{Array, IxDyn};
use rame::session::ort::{OrtBackend, OrtSessionConfig};
use rame::session::{InferSession, SessionBackend};
use rame::tensor::{TensorMap, TensorValue};

#[test]
fn ort_session_runs_named_f32_identity_model() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/identity.onnx");

    let mut session = OrtBackend::load(&model_path, OrtSessionConfig::default())?;
    let input = Array::from_shape_vec(IxDyn(&[3]), vec![1.0_f32, 2.0, 3.0]).unwrap();
    let mut inputs = TensorMap::new();
    inputs.insert("input".to_string(), TensorValue::F32(input.clone()));

    let mut outputs = session.run(inputs)?;
    let Some(TensorValue::F32(output)) = outputs.remove("output") else {
        panic!("expected f32 output named `output`");
    };

    assert_eq!(output, input);

    Ok(())
}
