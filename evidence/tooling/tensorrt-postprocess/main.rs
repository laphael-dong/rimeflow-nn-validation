#[path = "../../../src/postprocess.rs"]
mod production_postprocess;

use std::{fs, path::PathBuf};

use production_postprocess::{decode_yolo_output, nms};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let input_path = PathBuf::from(args.next().ok_or("raw output path missing")?);
    let width: u32 = args.next().ok_or("source width missing")?.to_string_lossy().parse()?;
    let height: u32 = args.next().ok_or("source height missing")?.to_string_lossy().parse()?;
    let output_path = PathBuf::from(args.next().ok_or("JSON output path missing")?);
    let bytes = fs::read(input_path)?;
    if bytes.len() != 84 * 8400 * size_of::<f32>() {
        return Err(format!("expected {} raw bytes, got {}", 84 * 8400 * 4, bytes.len()).into());
    }
    let raw: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    if raw.iter().any(|value| !value.is_finite()) {
        return Err("raw output contains a non-finite value".into());
    }
    let scale = (640.0 / width as f32).min(640.0 / height as f32);
    let pad_x = (640.0 - width as f32 * scale) / 2.0;
    let pad_y = (640.0 - height as f32 * scale) / 2.0;
    let detections = nms(&decode_yolo_output(&raw, width, height, scale, pad_x, pad_y), 0.45);
    let body = detections
        .iter()
        .map(|item| format!("  {{\"classId\":{},\"score\":{},\"bbox\":[{},{},{},{}]}}", item.class_id, item.score, item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3]))
        .collect::<Vec<_>>()
        .join(",\n");
    fs::write(output_path, format!("[\n{}\n]\n", body))?;
    Ok(())
}
