# rimeflow-yolov8n

YOLOv8n object detection — YOLO output decoding + Non-Maximum Suppression (NMS) in pure Rust.

## What it does

- Decodes YOLOv8 raw output tensor `[1, 84, 8400]` into detection bounding boxes
- Applies letterbox inverse mapping to convert model coordinates back to source image coordinates
- Runs NMS (Non-Maximum Suppression) to filter overlapping detections
- Supports all 80 COCO classes

## Usage

```rust
use rimeflow_yolov8n::postprocess::{decode_yolo_output, nms};

let detections = decode_yolo_output(&raw_output, src_w, src_h, scale, pad_x, pad_y);
let filtered = nms(&detections, 0.45);
```

## Model

The `models/yolov8n.onnx` file (~13MB) is the [Ultralytics YOLOv8n](https://github.com/ultralytics/ultralytics) model exported to ONNX format.

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx simplify=True opset=17
```

## License

MIT
