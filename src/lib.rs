//! YOLOv8n operator crate for the rimeflow family.
//!
//! Since **v0.2**, this crate is a thin operator-specialization layer over
//! [`rimeflow-onnx-base`](https://github.com/caozisheng/rimeflow-onnx-base):
//! all cross-target infrastructure (wgpu preprocess pipeline, letterbox math,
//! `pykeio/ort` v2 session + EP fan-out, `onnxruntime-web` inline_js bridge)
//! lives in the base crate; this crate only owns YOLOv8n's WGSL, ONNX weights,
//! and the model-specific `Detection` + decode + NMS in [`postprocess`].
//!
//! # Consumer API (unchanged from v0.1.x)
//!
//! ```ignore
//! use rimeflow_yolov8n::{
//!     preprocess::{LetterboxParams, PreprocessPipeline},
//!     postprocess::{decode_yolo_output, nms},
//!     MODEL_URL, MODEL_BYTES, DST_SIZE, INPUT_SHAPE,
//! };
//!
//! let pipeline = PreprocessPipeline::new(
//!     &device, DST_SIZE, PreprocessPipeline::default_buffer_usage(),
//! );
//! // ... same LetterboxParams / dispatch / postprocess as before.
//! ```
//!
//! Rimecut consumers see the same three-arg [`preprocess::PreprocessPipeline::new`]
//! signature, the same [`preprocess::LetterboxParams::compute`], and the same
//! [`postprocess::Detection`] as v0.1.x. Under the hood
//! `PreprocessPipeline::new` forwards to
//! `rimeflow_onnx_base::PreprocessPipeline::new(device, dst_size, usage, PREPROCESS_WGSL)`.
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
//! Full design context: `docs/rimecut-feature-dev-rules.md` §16 in the
//! RimeCut monorepo (the base-crate split design).

pub mod postprocess;

pub mod preprocess {
    //! YOLOv8n preprocess — thin wrapper over
    //! [`rimeflow_onnx_base::PreprocessPipeline`] that binds this crate's
    //! [`PREPROCESS_WGSL`] into the pipeline constructor.
    //!
    //! Consumers use the wrapper's 3-argument [`PreprocessPipeline::new`]
    //! (device, dst_size, usage); the WGSL is supplied internally so the
    //! v0.1.x call-site continues to compile.

    /// Single source of truth for the YOLOv8n preprocess shader.
    ///
    /// Injected into the wasm inline_js block by `build.rs` (via
    /// [`rimeflow_onnx_base::build_helper::generate_extern_block`]) so the
    /// canvas transition path (`ort_detect`) uses the same shader as the
    /// Rust wgpu path.
    pub const PREPROCESS_WGSL: &str = include_str!("../shaders/preprocess.wgsl");

    // Re-export letterbox math + uniform verbatim.
    pub use rimeflow_onnx_base::{LetterboxParams, PreprocessUniform};

    /// YOLOv8n preprocess pipeline — thin wrapper.
    ///
    /// Signature-compatible with v0.1.x
    /// (`PreprocessPipeline::new(&device, dst_size, usage)`). The WGSL is
    /// bound internally from [`PREPROCESS_WGSL`].
    pub struct PreprocessPipeline {
        inner: rimeflow_onnx_base::PreprocessPipeline,
    }

    impl PreprocessPipeline {
        /// Create the pipeline. See [v0.1.x docs][`rimeflow_onnx_base::PreprocessPipeline::new`]
        /// for `dst_buffer_usage` semantics.
        pub fn new(
            device: &wgpu::Device,
            dst_size: u32,
            dst_buffer_usage: wgpu::BufferUsages,
        ) -> Self {
            Self {
                inner: rimeflow_onnx_base::PreprocessPipeline::new(
                    device, dst_size, dst_buffer_usage, PREPROCESS_WGSL,
                ),
            }
        }

        /// Permissive buffer-usage default. See
        /// [`rimeflow_onnx_base::PreprocessPipeline::default_buffer_usage`].
        pub fn default_buffer_usage() -> wgpu::BufferUsages {
            rimeflow_onnx_base::PreprocessPipeline::default_buffer_usage()
        }

        /// Dispatch preprocess for one frame. See
        /// [`rimeflow_onnx_base::PreprocessPipeline::dispatch`].
        pub fn dispatch(
            &self,
            device:     &wgpu::Device,
            queue:      &wgpu::Queue,
            encoder:    &mut wgpu::CommandEncoder,
            scene_view: &wgpu::TextureView,
            letterbox:  LetterboxParams,
        ) {
            self.inner.dispatch(device, queue, encoder, scene_view, letterbox)
        }

        /// Preprocess output buffer — hand this to the ORT EP.
        pub fn output(&self) -> &wgpu::Buffer { self.inner.output() }
        pub fn dst_size(&self) -> u32 { self.inner.dst_size() }
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "native"))]
pub mod native_ort {
    //! YOLOv8n native ORT backend — re-export of
    //! [`rimeflow_onnx_base::native_ort`] verbatim. The base crate's
    //! `NativeOrtBackend` is model-agnostic (`dst_size` is a runtime arg),
    //! so YOLOv8n uses it directly with no wrapping.
    pub use rimeflow_onnx_base::native_ort::{
        InferError, NativeOrtBackend, ResolvedEp,
    };
}

#[cfg(target_arch = "wasm32")]
pub mod ort_bridge {
    //! YOLOv8n wasm bridge to onnxruntime-web.
    //!
    //! Contains:
    //! - The `#[wasm_bindgen(inline_js = "...")]` extern block generated by
    //!   `build.rs` (via `rimeflow_onnx_base::build_helper`). Exposes
    //!   `capture_webgpu_device`, `ort_init`, `ort_run_gpu_buffer`,
    //!   `ort_run_cpu_slice`, `ort_detect`, `ort_release`.
    //! - Re-exports of the base's helpers (`extract_web_gpu_buffer`,
    //!   `get_output_f32`, `get_f64`).
    include!(concat!(env!("OUT_DIR"), "/ort_bridge_generated.rs"));

    pub use rimeflow_onnx_base::ort_bridge::{
        extract_web_gpu_buffer, get_f64, get_output_f32,
    };
}

/// Web fetch URL — served by consumers under `apps/web/public/models/`.
pub const MODEL_URL: &str = "/models/yolov8n.onnx";

/// Native embedded model bytes — enabled with `feature = "model-embedded"`.
#[cfg(feature = "model-embedded")]
pub const MODEL_BYTES: &[u8] = include_bytes!("../models/yolov8n.onnx");

/// Model input tile size — YOLOv8n was exported at 640×640 letterbox.
pub const DST_SIZE: u32 = 640;

/// Model input tensor shape `[batch, channels, H, W]`.
pub const INPUT_SHAPE: [i64; 4] = [1, 3, DST_SIZE as i64, DST_SIZE as i64];
