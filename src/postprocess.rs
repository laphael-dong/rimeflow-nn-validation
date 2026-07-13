#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: [f32; 4], // x1, y1, x2, y2 normalized 0..1
    pub score: f32,
    pub class_id: u32,
}

pub fn decode_yolo_output(
    raw: &[f32],
    src_w: u32,
    src_h: u32,
    scale: f32,
    pad_x: f32,
    pad_y: f32,
) -> Vec<Detection> {
    // YOLOv8 output: [1, 84, 8400] = [batch, 4+80classes, anchors]
    let num_classes: usize = 80;
    let num_boxes: usize = 8400;
    let expected = (4 + num_classes) * num_boxes;
    if raw.len() < expected {
        return Vec::new();
    }

    let mut detections = Vec::new();

    for i in 0..num_boxes {
        let cx = raw[0 * num_boxes + i];
        let cy = raw[1 * num_boxes + i];
        let w = raw[2 * num_boxes + i];
        let h = raw[3 * num_boxes + i];

        let mut max_score: f32 = 0.0;
        let mut max_class: u32 = 0;
        for c in 0..num_classes {
            let score = raw[(4 + c) * num_boxes + i];
            if score > max_score {
                max_score = score;
                max_class = c as u32;
            }
        }

        if max_score < 0.25 {
            continue;
        }

        // Letterbox inverse mapping: model coords → normalized source coords
        let x1 = ((cx - w / 2.0) - pad_x) / (src_w as f32 * scale);
        let y1 = ((cy - h / 2.0) - pad_y) / (src_h as f32 * scale);
        let x2 = ((cx + w / 2.0) - pad_x) / (src_w as f32 * scale);
        let y2 = ((cy + h / 2.0) - pad_y) / (src_h as f32 * scale);

        detections.push(Detection {
            bbox: [
                x1.clamp(0.0, 1.0),
                y1.clamp(0.0, 1.0),
                x2.clamp(0.0, 1.0),
                y2.clamp(0.0, 1.0),
            ],
            score: max_score,
            class_id: max_class,
        });
    }

    detections
}

pub fn nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    let mut sorted: Vec<_> = detections.to_vec();
    sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = Vec::new();
    let mut suppressed = vec![false; sorted.len()];

    for i in 0..sorted.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(sorted[i].clone());
        for j in (i + 1)..sorted.len() {
            if suppressed[j] {
                continue;
            }
            if iou(&sorted[i].bbox, &sorted[j].bbox) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}

pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

