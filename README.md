<div align="center">

# rame

![CI](https://github.com/bndos/rame/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/github/license/bndos/rame)

Model-aware inference runtime.

</div>

`rame` provides high-level Rust APIs for model families with maintained runtime
integrations. It hides preprocessing, tensor names, inference sessions, and
postprocessing behind model-specific interfaces.

Execution is delegated to engines such as ONNX Runtime. `rame` focuses on the
runtime layer around those engines: artifacts, typed inputs and outputs,
benchmarking, and eventually serving.

## Architecture

```text
source -> artifact -> processor -> session -> decoder -> result
```

- **source**: resolves model files (HuggingFace or local path).
- **artifact**: export-specific recipe: model file, tensor names, preprocessing,
  backend session configuration, processor, and decoder.
- **processor**: converts user inputs into named backend tensors.
- **session**: runs the inference backend. ONNX Runtime is the first backend.
- **decoder**: converts backend output tensors into typed model results.
- **pipeline**: runtime composition of processor, session, and decoder.

## Rust Usage

```rust
use rame::layout::LayoutModel;
use rame::models::pp_doclayout::plus::{self, PpDocLayoutPlus};
use rame::sources::HuggingFace;

let source = HuggingFace::new()?.model("PaddlePaddle/PP-DocLayout_plus-L_onnx");

let mut model = PpDocLayoutPlus::builder()
    .source(source)
    .artifact(plus::onnx::Artifact::default())
    .build()?;

let result = model.detect_layout(&image)?;
```


## Supported Models

| Family | Task | Artifact |
| --- | --- | --- |
| PaddleOCR PP-DocLayout Plus | Layout detection | ONNX |
