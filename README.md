# rimeflow-yolov8n

YOLOv8n object detection for Rust + WASM — output decoding, NMS, and ONNX Runtime Web bridge.

## What it does

- **postprocess**: Decodes YOLOv8 raw output tensor `[1, 84, 8400]` into bounding boxes, applies letterbox inverse mapping, runs NMS — pure Rust, no external deps
- **ort_bridge**: WASM-only bridge to [onnxruntime-web](https://onnxruntime.ai/) via `wasm_bindgen(inline_js)` — loads ORT at runtime, runs inference with WebGPU EP, includes GPU preprocessing compute shader and CPU fallback

## Architecture

```
┌─ Rust/WASM ──────────────────────────────────────────┐
│                                                       │
│  postprocess.rs       pure Rust, all targets          │
│    decode_yolo_output()  tensor → Detection[]         │
│    nms()                 filter overlapping boxes      │
│                                                       │
│  ort_bridge.rs        wasm32 only (inline_js)         │
│    capture_webgpu_device()  intercept GPUDevice       │
│    ort_init(model_url)      load ORT + create session │
│    ort_detect(canvas)       preprocess + inference    │
│    ort_release()            cleanup                   │
│                                                       │
└───────────────────────┬───────────────────────────────┘
                        │ wasm_bindgen(inline_js)
                        ▼
┌─ JS (embedded in WASM glue) ─────────────────────────┐
│  onnxruntime-web loaded from /ort/ort.min.js          │
│  WebGPU EP for GPU-accelerated inference              │
│  CPU fallback via OffscreenCanvas for WebGL           │
└───────────────────────────────────────────────────────┘
```

The `inline_js` approach (inspired by [eidon](https://github.com/FrancescoCitti/eidon)) embeds ~80 lines of JS directly in the WASM glue code. No separate JS files or npm imports needed — just `<script src="/ort/ort.min.js">` on the page.

## Usage

```rust
use rimeflow_yolov8n::postprocess::{decode_yolo_output, nms};

// Decode + filter
let detections = decode_yolo_output(&raw_output, src_w, src_h, scale, pad_x, pad_y);
let filtered = nms(&detections, 0.45);
```

```rust
// WASM only: ORT inference bridge
use rimeflow_yolov8n::ort_bridge;

ort_bridge::capture_webgpu_device();  // before GPU init
ort_bridge::ort_init("/models/yolov8n.onnx").await?;
let result = ort_bridge::ort_detect(&canvas).await?;
```

## Model

The `models/yolov8n.onnx` file (~13MB) is [Ultralytics YOLOv8n](https://github.com/ultralytics/ultralytics) exported to ONNX.

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx simplify=True opset=17
```

80 COCO classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

## License

MIT
