//! YOLOv8n build.rs — delegates everything to
//! `rimeflow_onnx_base::build_helper::generate_extern_block`.
//!
//! That helper reads `shaders/preprocess.wgsl` + the base crate's `template.js`,
//! substitutes the sentinels, and writes `${OUT_DIR}/ort_bridge_generated.rs`.
//! Our `src/lib.rs` picks it up via `include!`.

use std::path::Path;

use rimeflow_onnx_base::build_helper::{generate_extern_block, BridgeConfig};

fn main() {
    generate_extern_block(&BridgeConfig {
        wgsl_path: Path::new("shaders/preprocess.wgsl"),
        dst_size:  640,
    })
    .expect("rimeflow-onnx-base build_helper failed");
}
