#[cfg(test)]
#[allow(clippy::erasing_op, clippy::identity_op)]
#[path = "../../../../src/postprocess.rs"]
mod production_postprocess;

#[cfg(test)]
mod seam;

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use serde_json::Value;

    use super::production_postprocess::{decode_yolo_output, nms, Detection};
    use super::seam::{
        compare_raw_golden, normalize_output, prepare_input, resolve_postprocess_plan,
        validate_golden_summary, ContractError, GoldenSummary, GoldenTolerance, LogicalImage,
        LogicalRole, LogicalTensor, Quantization, RuntimeTensor, TensorDType, TensorData,
        TensorLayout, TensorSpec,
    };

    const ATTRIBUTES: usize = 84;
    const ANCHORS: usize = 8400;
    const MODEL_SIZE: f32 = 640.0;
    const EPSILON: f32 = 1.0e-6;

    macro_rules! rfb_val_red_test {
        ($id:literal, $name:ident, $body:block) => {
            #[test]
            fn $name() {
                fn run() -> Result<(), ContractError> $body
                let result = run();
                if let Err(error) = result {
                    panic!(
                        "{}: target_assertion: expected contract behavior; actual={}",
                        $id, error
                    );
                }
            }
        };
    }

    fn root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }

    fn read_json(path: &str) -> Value {
        serde_json::from_slice(&fs::read(root().join(path)).unwrap()).unwrap()
    }

    fn shape(value: &Value) -> Vec<usize> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|dimension| dimension.as_u64().unwrap() as usize)
            .collect()
    }

    fn frozen_tolerance() -> GoldenTolerance {
        let web = read_json("evidence/golden/web-reference.json");
        let tolerances = &web["tolerances"];
        GoldenTolerance {
            raw_absolute: tolerances["rawTensorAbsolute"]
                .as_f64()
                .map(|value| value as f32),
            raw_relative: tolerances["rawTensorRelative"]
                .as_f64()
                .map(|value| value as f32),
            confidence_absolute: tolerances["confidenceAbsolute"]
                .as_f64()
                .map(|value| value as f32),
            box_iou_minimum: tolerances["boxIouMinimum"]
                .as_f64()
                .map(|value| value as f32),
        }
    }

    fn onnx_output_spec() -> TensorSpec {
        let contract = read_json("evidence/model/model-contract.json");
        let output = &contract["model"]["output"];
        TensorSpec {
            role: Some(LogicalRole::Detections),
            runtime_name: output["runtimeName"].as_str().unwrap().to_owned(),
            runtime_index: output["index"].as_u64().unwrap() as usize,
            shape: shape(&output["shape"]),
            layout: TensorLayout::AttributesAnchors,
            dtype: TensorDType::F32,
            quantization: None,
            coordinate_scale: 1.0,
            nms_fused: output["semantics"]["nmsFused"].as_bool().unwrap(),
        }
    }

    fn litert_output_spec() -> TensorSpec {
        let manifest = read_json("evidence/conversions/litert-artifact-manifest.json");
        let output = &manifest["ioContract"]["output"];
        assert_eq!(manifest["ioContract"]["outputRole"], "detections");
        TensorSpec {
            role: Some(LogicalRole::Detections),
            runtime_name: output["name"].as_str().unwrap().to_owned(),
            runtime_index: output["index"].as_u64().unwrap() as usize,
            shape: shape(&output["shape"]),
            layout: TensorLayout::AttributesAnchors,
            dtype: TensorDType::F32,
            quantization: None,
            coordinate_scale: MODEL_SIZE,
            nms_fused: false,
        }
    }

    fn input_spec(layout: TensorLayout, dtype: TensorDType) -> TensorSpec {
        let shape = match layout {
            TensorLayout::Nchw => vec![1, 3, 2, 2],
            TensorLayout::Nhwc => vec![1, 2, 2, 3],
            TensorLayout::AttributesAnchors => unreachable!(),
        };
        TensorSpec {
            role: Some(LogicalRole::Image),
            runtime_name: "images".to_owned(),
            runtime_index: 0,
            shape,
            layout,
            dtype,
            quantization: None,
            coordinate_scale: 1.0,
            nms_fused: false,
        }
    }

    fn raw_fixture(id: &str) -> Value {
        read_json(&format!("evidence/fixtures/raw/{id}.json"))
    }

    fn sparse_raw(fixture: &Value) -> Vec<f32> {
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
        raw
    }

    fn runtime_tensor(spec: &TensorSpec, values: Vec<f32>) -> RuntimeTensor {
        RuntimeTensor {
            name: spec.runtime_name.clone(),
            index: spec.runtime_index,
            shape: spec.shape.clone(),
            data: TensorData::F32(values),
        }
    }

    fn geometry(fixture: &Value) -> (u32, u32, f32, f32, f32) {
        let width = fixture["geometry"]["width"].as_u64().unwrap() as u32;
        let height = fixture["geometry"]["height"].as_u64().unwrap() as u32;
        let scale = (MODEL_SIZE / width as f32).min(MODEL_SIZE / height as f32);
        let pad_x = (MODEL_SIZE - width as f32 * scale) / 2.0;
        let pad_y = (MODEL_SIZE - height as f32 * scale) / 2.0;
        (width, height, scale, pad_x, pad_y)
    }

    fn detections_match(actual: &[Detection], expected: &[Value]) {
        assert_eq!(actual.len(), expected.len(), "detection count");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.class_id,
                expected["classId"].as_u64().unwrap() as u32,
                "detection {index} class"
            );
            assert!(
                (actual.score - expected["score"].as_f64().unwrap() as f32).abs() <= EPSILON,
                "detection {index} score"
            );
            for axis in 0..4 {
                assert!(
                    (actual.bbox[axis] - expected["bbox"][axis].as_f64().unwrap() as f32).abs()
                        <= EPSILON,
                    "detection {index} bbox axis {axis}"
                );
            }
        }
    }

    fn decode_and_nms(fixture: &Value, values: &[f32]) -> Vec<Detection> {
        let (width, height, scale, pad_x, pad_y) = geometry(fixture);
        let decoded = decode_yolo_output(values, width, height, scale, pad_x, pad_y);
        detections_match(
            &decoded,
            fixture["expectedPostprocess"]["decoded"]
                .as_array()
                .unwrap(),
        );
        let kept = nms(
            &decoded,
            fixture["expectedPostprocess"]["nmsIouThreshold"]
                .as_f64()
                .unwrap() as f32,
        );
        detections_match(
            &kept,
            fixture["expectedPostprocess"]["nms"].as_array().unwrap(),
        );
        kept
    }

    fn ppm_rgb(path: &str) -> LogicalImage {
        let bytes = fs::read(root().join(path)).unwrap();
        let mut cursor = 0usize;
        let mut token = || {
            while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            let start = cursor;
            while cursor < bytes.len() && !bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            std::str::from_utf8(&bytes[start..cursor])
                .unwrap()
                .to_owned()
        };
        assert_eq!(token(), "P6");
        let width = token().parse::<usize>().unwrap();
        let height = token().parse::<usize>().unwrap();
        assert_eq!(token(), "255");
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        let rgb = bytes[cursor..].to_vec();
        assert_eq!(rgb.len(), width * height * 3);
        LogicalImage { width, height, rgb }
    }

    fn two_by_two_sample(path: &str) -> LogicalImage {
        let image = ppm_rgb(path);
        let mut rgb = Vec::with_capacity(12);
        for y in 0..2 {
            let start = y * image.width * 3;
            rgb.extend_from_slice(&image.rgb[start..start + 6]);
        }
        LogicalImage {
            width: 2,
            height: 2,
            rgb,
        }
    }

    fn expect_error(
        result: Result<impl Sized, ContractError>,
        expected: ContractError,
    ) -> Result<(), ContractError> {
        match result {
            Err(error) if error == expected => Ok(()),
            Err(error) => Err(error),
            Ok(_) => panic!("expected contract error {expected}"),
        }
    }

    rfb_val_red_test!(
        "RFB-VAL-IO-001",
        rfb_val_io_001_layout_and_dtype_preserve_logical_image,
        {
            let image = two_by_two_sample("evidence/fixtures/images/multi-class.ppm");
            let nchw = prepare_input(&image, &input_spec(TensorLayout::Nchw, TensorDType::F32))?;
            let nhwc = prepare_input(&image, &input_spec(TensorLayout::Nhwc, TensorDType::F32))?;
            assert_eq!(nchw.shape, vec![1, 3, 2, 2]);
            assert_eq!(nhwc.shape, vec![1, 2, 2, 3]);
            assert_ne!(
                nchw.data, nhwc.data,
                "layout-specific byte order must differ"
            );
            Ok(())
        }
    );

    rfb_val_red_test!(
        "RFB-VAL-IO-002",
        rfb_val_io_002_invalid_quantization_is_rejected,
        {
            let image = two_by_two_sample("evidence/fixtures/images/multi-class.ppm");
            let mut spec = input_spec(TensorLayout::Nhwc, TensorDType::U8);
            spec.quantization = Some(Quantization {
                scale: 0.0,
                zero_point: 0,
            });
            expect_error(
                prepare_input(&image, &spec),
                ContractError::InvalidQuantization,
            )
        }
    );

    rfb_val_red_test!(
        "RFB-VAL-ROLE-001",
        rfb_val_role_001_renamed_output_maps_to_detections,
        {
            let fixture = raw_fixture("single-target");
            let spec = litert_output_spec();
            let mut normalized = sparse_raw(&fixture);
            for attribute in 0..4 {
                for value in &mut normalized[attribute * ANCHORS..(attribute + 1) * ANCHORS] {
                    *value /= MODEL_SIZE;
                }
            }
            let logical = normalize_output(&spec, &runtime_tensor(&spec, normalized))?;
            assert_eq!(logical.role, LogicalRole::Detections);
            decode_and_nms(&fixture, &logical.values);
            Ok(())
        }
    );

    rfb_val_red_test!(
        "RFB-VAL-ROLE-002",
        rfb_val_role_002_missing_output_role_is_rejected,
        {
            let fixture = raw_fixture("single-target");
            let mut spec = onnx_output_spec();
            spec.role = None;
            expect_error(
                normalize_output(&spec, &runtime_tensor(&spec, sparse_raw(&fixture))),
                ContractError::MissingLogicalRole,
            )
        }
    );

    rfb_val_red_test!(
        "RFB-VAL-GOLDEN-001",
        rfb_val_golden_001_all_raw_and_decoded_fixtures_match,
        {
            let manifest = read_json("evidence/fixtures/manifest.json");
            let fixtures = manifest["rawTensorFixtures"].as_array().unwrap();
            assert_eq!(fixtures.len(), 6);
            for entry in fixtures {
                let fixture = read_json(entry["path"].as_str().unwrap());
                let spec = onnx_output_spec();
                let logical =
                    normalize_output(&spec, &runtime_tensor(&spec, sparse_raw(&fixture)))?;
                decode_and_nms(&fixture, &logical.values);
            }
            Ok(())
        }
    );

    rfb_val_red_test!(
        "RFB-VAL-GOLDEN-002",
        rfb_val_golden_002_out_of_tolerance_output_is_rejected,
        {
            let fixture = raw_fixture("single-target");
            let values = sparse_raw(&fixture);
            let reference = LogicalTensor {
                role: LogicalRole::Detections,
                shape: vec![1, ATTRIBUTES, ANCHORS],
                values: values.clone(),
            };
            let mut candidate = reference.clone();
            candidate.values[(4 + 16) * ANCHORS + 1] += 0.01;
            expect_error(
                compare_raw_golden(&reference, &candidate, &frozen_tolerance()),
                ContractError::GoldenMismatch,
            )
        }
    );

    rfb_val_red_test!(
        "RFB-VAL-POST-001",
        rfb_val_post_001_preprocessing_responsibility_is_applied_once,
        {
            let coreml = read_json("evidence/conversions/coreml-artifact-manifest.json");
            let input = &coreml["spec"]["input"];
            assert_eq!(input["name"], "image");
            assert_eq!(
                input["layout"],
                "RGB image feature; ML Program function tensor NCHW"
            );
            assert_eq!(
                coreml["spec"]["preprocessing"]["fused"]
                    .as_array()
                    .unwrap()
                    .len(),
                2
            );
            assert_eq!(
                coreml["spec"]["preprocessing"]["notFused"]
                    .as_array()
                    .unwrap()
                    .len(),
                2
            );
            let image = two_by_two_sample("evidence/fixtures/images/multi-class.ppm");
            prepare_input(&image, &input_spec(TensorLayout::Nchw, TensorDType::F32))?;
            Ok(())
        }
    );

    rfb_val_red_test!(
        "RFB-VAL-POST-002",
        rfb_val_post_002_decode_and_nms_have_one_owner,
        {
            let fixture = raw_fixture("overlap-nms");
            let raw = sparse_raw(&fixture);
            let direct = decode_and_nms(&fixture, &raw);
            assert_eq!(direct.len(), 1, "production NMS must suppress the overlap");

            let operator_owned = onnx_output_spec();
            let plan = resolve_postprocess_plan(&operator_owned)?;
            assert!(plan.decode_in_operator);
            assert!(plan.threshold_in_operator);
            assert!(plan.nms_in_operator);

            let mut fused = operator_owned;
            fused.nms_fused = true;
            let fused_plan = resolve_postprocess_plan(&fused)?;
            assert!(!fused_plan.nms_in_operator, "fused NMS must not run twice");
            Ok(())
        }
    );

    rfb_val_red_test!(
        "RFB-VAL-DETERMINISM-001",
        rfb_val_determinism_001_three_logical_runs_are_identical,
        {
            let manifest = read_json("evidence/fixtures/manifest.json");
            for entry in manifest["rawTensorFixtures"].as_array().unwrap() {
                let fixture = read_json(entry["path"].as_str().unwrap());
                let spec = onnx_output_spec();
                let runtime = runtime_tensor(&spec, sparse_raw(&fixture));
                let mut signatures = Vec::new();
                for _ in 0..3 {
                    let logical = normalize_output(&spec, &runtime)?;
                    let detections = decode_and_nms(&fixture, &logical.values);
                    signatures.push(
                        detections
                            .iter()
                            .map(|detection| {
                                (
                                    detection.class_id,
                                    detection.score.to_bits(),
                                    detection.bbox.map(f32::to_bits),
                                )
                            })
                            .collect::<Vec<_>>(),
                    );
                }
                assert!(signatures.windows(2).all(|pair| pair[0] == pair[1]));
            }
            Ok(())
        }
    );

    rfb_val_red_test!(
        "RFB-VAL-FIXTURE-001",
        rfb_val_fixture_001_web_golden_covers_all_image_fixtures,
        {
            let fixture_manifest = read_json("evidence/fixtures/manifest.json");
            let web = read_json("evidence/golden/web-reference.json");
            let image_ids = fixture_manifest["images"]
                .as_array()
                .unwrap()
                .iter()
                .map(|entry| entry["id"].as_str().unwrap())
                .collect::<Vec<_>>();
            let web_fixtures = web["fixtures"].as_array().unwrap();
            assert_eq!(web_fixtures.len(), image_ids.len());
            for fixture in web_fixtures {
                assert!(image_ids.contains(&fixture["id"].as_str().unwrap()));
                assert_eq!(fixture["runs"].as_array().unwrap().len(), 3);
                let raw = &fixture["runs"][0]["rawTensor"];
                let summary = GoldenSummary {
                    role: Some(LogicalRole::Detections),
                    shape: vec![1, ATTRIBUTES, ANCHORS],
                    element_count: raw["elementCount"].as_u64().unwrap() as usize,
                    finite_count: raw["finiteCount"].as_u64().unwrap() as usize,
                    raw_digest: raw["sha256Float32Le"].as_str().unwrap().to_owned(),
                };
                validate_golden_summary(&summary, &frozen_tolerance())?;
                assert_eq!(fixture["determinism"]["allRawDigestsEqual"], true);
                assert_eq!(fixture["determinism"]["allDecodedEqual"], true);
            }
            Ok(())
        }
    );

    rfb_val_red_test!(
        "RFB-VAL-FIXTURE-002",
        rfb_val_fixture_002_missing_tolerance_is_rejected,
        {
            let web = read_json("evidence/golden/web-reference.json");
            let raw = &web["fixtures"][0]["runs"][0]["rawTensor"];
            let summary = GoldenSummary {
                role: Some(LogicalRole::Detections),
                shape: vec![1, ATTRIBUTES, ANCHORS],
                element_count: raw["elementCount"].as_u64().unwrap() as usize,
                finite_count: raw["finiteCount"].as_u64().unwrap() as usize,
                raw_digest: raw["sha256Float32Le"].as_str().unwrap().to_owned(),
            };
            let mut tolerance = frozen_tolerance();
            tolerance.raw_absolute = None;
            expect_error(
                validate_golden_summary(&summary, &tolerance),
                ContractError::MissingTolerance,
            )
        }
    );
}
