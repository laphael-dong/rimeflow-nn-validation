pub mod postprocess;

#[cfg(target_arch = "wasm32")]
pub mod ort_bridge;

pub const MODEL_URL: &str = "/models/yolov8n.onnx";
