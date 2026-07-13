// YOLOv8n input preprocess — letterbox resize + RGBA→NCHW normalize.
//
// Single source of truth (see r4 §3 contract).
//
// - Rust side: included via `include_str!` in src/preprocess.rs (used by native
//   backend and by the wasm main-line path).
// - JS side: injected by build.rs into the inline_js template as
//   `PREPROCESS_WGSL` at compile time (used only by the transition
//   `ort_detect(canvas)` API for open_quartz).
//
// Both sides run structurally identical compute passes: same bind group layout,
// same workgroup size, same dst layout. Letterbox parameter math is anchored
// separately in Rust (`LetterboxParams::compute`) and JS
// (`_computeLetterbox` in ort_bridge_template.js) — they MUST stay in sync.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

struct Params {
  src_w:    f32,
  src_h:    f32,
  dst_size: u32,
  _pad_a:   u32,
  scale:    f32,
  pad_x:    f32,
  pad_y:    f32,
  _pad_b:   f32,
}
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;
  let y = gid.y;
  if (x >= p.dst_size || y >= p.dst_size) { return; }

  let dst_f = f32(p.dst_size);
  let out_u = f32(x) / dst_f;
  let out_v = f32(y) / dst_f;
  // The scaled source occupies [pad_x, 1-pad_x] × [pad_y, 1-pad_y] in
  // destination-normalized space. Divide by the per-axis extent, not the
  // scalar `p.scale` (which only equals the extent on the square case).
  let region_u = 1.0 - 2.0 * p.pad_x;
  let region_v = 1.0 - 2.0 * p.pad_y;
  let in_u  = (out_u - p.pad_x) / region_u;
  let in_v  = (out_v - p.pad_y) / region_v;

  var pixel: vec4f;
  if (in_u < 0.0 || in_u > 1.0 || in_v < 0.0 || in_v > 1.0) {
    // Standard YOLO letterbox gray fill (0x72 / 255 ≈ 0.447).
    pixel = vec4f(0.447, 0.447, 0.447, 1.0);
  } else {
    pixel = textureSampleLevel(src, src_sampler, vec2f(in_u, in_v), 0.0);
  }

  // NCHW layout: R plane, then G plane, then B plane.
  let hw  = p.dst_size * p.dst_size;
  let idx = y * p.dst_size + x;
  dst[0u * hw + idx] = pixel.r;
  dst[1u * hw + idx] = pixel.g;
  dst[2u * hw + idx] = pixel.b;
}
