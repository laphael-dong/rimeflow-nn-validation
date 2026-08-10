#[cfg(test)]
#[path = "../../../../src/postprocess.rs"]
mod production_postprocess;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::production_postprocess::{decode_yolo_output, nms, Detection};
    use serde_json::Value;

    const ATTRIBUTES: usize = 84;
    const ANCHORS: usize = 8400;
    const EPSILON: f32 = 1.0e-6;

    fn root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }

    fn assert_close(actual: f32, expected: f32, context: &str) {
        assert!(
            (actual - expected).abs() <= EPSILON,
            "{context}: expected {expected}, got {actual}"
        );
    }

    fn assert_detections(actual: &[Detection], expected: &[Value], context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context}: detection count");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.class_id,
                expected["classId"].as_u64().unwrap() as u32,
                "{context}[{index}]: class"
            );
            assert_close(
                actual.score,
                expected["score"].as_f64().unwrap() as f32,
                &format!("{context}[{index}]: score"),
            );
            let bbox = expected["bbox"].as_array().unwrap();
            for (axis, expected) in bbox.iter().enumerate() {
                assert_close(
                    actual.bbox[axis],
                    expected.as_f64().unwrap() as f32,
                    &format!("{context}[{index}].bbox[{axis}]"),
                );
            }
        }
    }

    #[test]
    fn task1_raw_golden_uses_production_decode_and_nms() {
        let root = root();
        let manifest: Value = serde_json::from_slice(
            &fs::read(root.join("evidence/fixtures/manifest.json")).unwrap(),
        )
        .unwrap();
        let fixtures = manifest["rawTensorFixtures"].as_array().unwrap();
        assert_eq!(fixtures.len(), 6, "任务 1 必须保留六类 raw fixture");

        for entry in fixtures {
            assert_eq!(entry["kind"], "raw-tensor");
            assert!(entry["sourceImage"].is_null());
            let fixture: Value = serde_json::from_slice(
                &fs::read(root.join(entry["path"].as_str().unwrap())).unwrap(),
            )
            .unwrap();
            assert_eq!(fixture["kind"], "manually-constructed-raw-tensor");
            assert!(fixture["sourceImage"].is_null());
            assert_eq!(fixture["modelShape"], serde_json::json!([1, 84, 8400]));

            let mut raw = vec![0.0f32; ATTRIBUTES * ANCHORS];
            for anchor in fixture["anchors"].as_array().unwrap() {
                let index = anchor["anchor"].as_u64().unwrap() as usize;
                let class_id = anchor["classId"].as_u64().unwrap() as usize;
                raw[index] = anchor["cx"].as_f64().unwrap() as f32;
                raw[ANCHORS + index] = anchor["cy"].as_f64().unwrap() as f32;
                raw[2 * ANCHORS + index] = anchor["width"].as_f64().unwrap() as f32;
                raw[3 * ANCHORS + index] = anchor["height"].as_f64().unwrap() as f32;
                raw[(4 + class_id) * ANCHORS + index] = anchor["score"].as_f64().unwrap() as f32;
            }

            let width = fixture["geometry"]["width"].as_u64().unwrap() as u32;
            let height = fixture["geometry"]["height"].as_u64().unwrap() as u32;
            let scale = (640.0 / width as f32).min(640.0 / height as f32);
            let pad_x = (640.0 - width as f32 * scale) / 2.0;
            let pad_y = (640.0 - height as f32 * scale) / 2.0;
            let decoded = decode_yolo_output(&raw, width, height, scale, pad_x, pad_y);
            let id = entry["id"].as_str().unwrap();
            assert_detections(
                &decoded,
                fixture["expectedPostprocess"]["decoded"]
                    .as_array()
                    .unwrap(),
                &format!("{id}.decoded"),
            );
            let kept = nms(
                &decoded,
                fixture["expectedPostprocess"]["nmsIouThreshold"]
                    .as_f64()
                    .unwrap() as f32,
            );
            assert_detections(
                &kept,
                fixture["expectedPostprocess"]["nms"].as_array().unwrap(),
                &format!("{id}.nms"),
            );
        }
    }
}
