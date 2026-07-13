//! YOLOv8n inference building blocks.
//!
//! # Modules
//!
//! - [`postprocess`] — pure Rust `Detection` + `decode_yolo_output` + `nms`.
//! - [`preprocess`] — cross-target letterbox + NCHW normalize (Rust wgpu
//!   `PreprocessPipeline`, single-source WGSL).
//! - [`ort_bridge`] — wasm-only, `wasm_bindgen(inline_js)` glue to
//!   onnxruntime-web (WebGPU EP). Exposes the main-line
//!   `ort_run_gpu_buffer(gpu_buffer, dims)` API plus the transition
//!   `ort_detect(canvas)` API for [open_quartz](https://github.com/caozisheng/open_quartz).
//! - [`native_ort`] — non-wasm, `feature = "native"`, wraps
//!   [`pykeio/ort`](https://github.com/pykeio/ort) v2 with EP fan-out for
//!   CoreML/DirectML/CUDA/TensorRT/NNAPI/QNN/OpenVINO and a CPU fallback.
//!
//! # Model
//!
//! The model file is [Ultralytics YOLOv8n](https://github.com/ultralytics/ultralytics)
//! exported to ONNX (`yolo export model=yolov8n.pt format=onnx simplify=True opset=17`),
//! ~13 MB. It is served over HTTP at `MODEL_URL` (for wasm consumers) and
//! optionally embedded via `feature = "model-embedded"` as `MODEL_BYTES`
//! (for native consumers that don't want a separate file to ship).
//!
//! # Design context
//!
//! This crate is a rimeflow-family operator following the RimeCut NN
//! development template (`rimecut-feature-dev-onnx-development-rules.md`),
//! revision r4. Two consumers are supported:
//!
//! 1. **RimeCut main-line** (`apps/web` + `apps/tauri`) — uses `preprocess`
//!    + `ort_run_gpu_buffer` (wasm) or `native_ort` (native); Tier A / Tier B
//!    zero-copy in most cases.
//! 2. **open_quartz** — canvas-based debug/editor tool — uses `ort_detect(canvas)`
//!    (Tier C, one extra `copyExternalImageToTexture` hop).
//!
//! Both consumers share the same model, WGSL, letterbox math, decode, and NMS.
//! The wasm dual API is a transition arrangement; when open_quartz gains a
//! wgpu-based frame source, `ort_detect(canvas)` is retired (r4 §15 D10).

pub mod postprocess;
pub mod preprocess;

#[cfg(target_arch = "wasm32")]
pub mod ort_bridge;

#[cfg(all(not(target_arch = "wasm32"), feature = "native"))]
pub mod native_ort;

/// Web fetch URL — served by consumers under `apps/web/public/models/`.
pub const MODEL_URL: &str = "/models/yolov8n.onnx";

/// Native embedded model bytes — enabled with `feature = "model-embedded"`.
#[cfg(feature = "model-embedded")]
pub const MODEL_BYTES: &[u8] = include_bytes!("../models/yolov8n.onnx");

/// Model input tile size — YOLOv8n was exported at 640×640 letterbox.
pub const DST_SIZE: u32 = 640;

/// Model input tensor shape `[batch, channels, H, W]`.
pub const INPUT_SHAPE: [i64; 4] = [1, 3, DST_SIZE as i64, DST_SIZE as i64];
