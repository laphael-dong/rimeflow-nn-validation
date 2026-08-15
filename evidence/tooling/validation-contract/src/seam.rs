//! Validation-side adapters for the frozen runtime manifest.
//!
//! The adapter deliberately receives a logical role from the manifest. Runtime
//! tensor names and indexes are retained as diagnostics only; they are not used
//! to infer the image or detections role.

#![allow(dead_code)]

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalRole {
    Image,
    Detections,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    Nchw,
    Nhwc,
    AttributesAnchors,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F32,
    U8,
    I8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantization {
    pub scale: f32,
    pub zero_point: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorSpec {
    pub role: Option<LogicalRole>,
    pub runtime_name: String,
    pub runtime_index: usize,
    pub shape: Vec<usize>,
    pub layout: TensorLayout,
    pub dtype: TensorDType,
    pub quantization: Option<Quantization>,
    pub coordinate_scale: f32,
    pub nms_fused: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    U8(Vec<u8>),
    I8(Vec<i8>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeTensor {
    pub name: String,
    pub index: usize,
    pub shape: Vec<usize>,
    pub data: TensorData,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalTensor {
    pub role: LogicalRole,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalImage {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GoldenTolerance {
    pub raw_absolute: Option<f32>,
    pub raw_relative: Option<f32>,
    pub confidence_absolute: Option<f32>,
    pub box_iou_minimum: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GoldenSummary {
    pub role: Option<LogicalRole>,
    pub shape: Vec<usize>,
    pub element_count: usize,
    pub finite_count: usize,
    pub raw_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostprocessPlan {
    pub decode_in_operator: bool,
    pub threshold_in_operator: bool,
    pub nms_in_operator: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContractError {
    MissingLogicalRole,
    UnsupportedInputContract,
    InvalidQuantization,
    GoldenMismatch,
    MissingTolerance,
    TensorShapeMismatch,
    TensorDataMismatch,
    NonFiniteTensor,
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingLogicalRole => formatter.write_str("missing_logical_role"),
            Self::UnsupportedInputContract => formatter.write_str("unsupported_input_contract"),
            Self::InvalidQuantization => formatter.write_str("invalid_quantization"),
            Self::GoldenMismatch => formatter.write_str("golden_mismatch"),
            Self::MissingTolerance => formatter.write_str("missing_tolerance"),
            Self::TensorShapeMismatch => formatter.write_str("tensor_shape_mismatch"),
            Self::TensorDataMismatch => formatter.write_str("tensor_data_mismatch"),
            Self::NonFiniteTensor => formatter.write_str("non_finite_tensor"),
        }
    }
}

pub fn prepare_input(
    image: &LogicalImage,
    spec: &TensorSpec,
) -> Result<RuntimeTensor, ContractError> {
    if spec.role != Some(LogicalRole::Image)
        || !matches!(spec.layout, TensorLayout::Nchw | TensorLayout::Nhwc)
        || image.width == 0
        || image.height == 0
        || image.rgb.len() != image.width * image.height * 3
        || spec.shape.len() != 4
    {
        return Err(ContractError::UnsupportedInputContract);
    }
    let expected_shape = match spec.layout {
        TensorLayout::Nchw => vec![1, 3, image.height, image.width],
        TensorLayout::Nhwc => vec![1, image.height, image.width, 3],
        TensorLayout::AttributesAnchors => unreachable!(),
    };
    if spec.shape != expected_shape {
        return Err(ContractError::TensorShapeMismatch);
    }
    if let Some(quantization) = spec.quantization {
        if !quantization.scale.is_finite() || quantization.scale <= 0.0 {
            return Err(ContractError::InvalidQuantization);
        }
    }

    let ordered = ordered_rgb(image, spec.layout);
    let data = match spec.dtype {
        TensorDType::F32 => {
            if spec.quantization.is_some() {
                return Err(ContractError::UnsupportedInputContract);
            }
            TensorData::F32(
                ordered
                    .into_iter()
                    .map(|value| value as f32 / 255.0)
                    .collect(),
            )
        }
        TensorDType::U8 => {
            let quantization = spec
                .quantization
                .ok_or(ContractError::InvalidQuantization)?;
            TensorData::U8(
                ordered
                    .into_iter()
                    .map(|value| {
                        ((value as f32 / 255.0) / quantization.scale).round() as i32
                            + quantization.zero_point
                    })
                    .map(|value| value.clamp(0, 255) as u8)
                    .collect(),
            )
        }
        TensorDType::I8 => {
            let quantization = spec
                .quantization
                .ok_or(ContractError::InvalidQuantization)?;
            TensorData::I8(
                ordered
                    .into_iter()
                    .map(|value| {
                        ((value as f32 / 255.0) / quantization.scale).round() as i32
                            + quantization.zero_point
                    })
                    .map(|value| value.clamp(-128, 127) as i8)
                    .collect(),
            )
        }
    };
    Ok(RuntimeTensor {
        name: spec.runtime_name.clone(),
        index: spec.runtime_index,
        shape: spec.shape.clone(),
        data,
    })
}

pub fn normalize_output(
    spec: &TensorSpec,
    runtime_tensor: &RuntimeTensor,
) -> Result<LogicalTensor, ContractError> {
    if spec.role != Some(LogicalRole::Detections) {
        return Err(ContractError::MissingLogicalRole);
    }
    if spec.layout != TensorLayout::AttributesAnchors || spec.shape != runtime_tensor.shape {
        return Err(ContractError::TensorShapeMismatch);
    }
    let expected = product(&spec.shape).ok_or(ContractError::TensorShapeMismatch)?;
    let mut values = match (&spec.dtype, &runtime_tensor.data) {
        (TensorDType::F32, TensorData::F32(values)) => values.clone(),
        (TensorDType::U8, TensorData::U8(values)) => dequantize_u8(values, spec.quantization)?,
        (TensorDType::I8, TensorData::I8(values)) => dequantize_i8(values, spec.quantization)?,
        _ => return Err(ContractError::TensorDataMismatch),
    };
    if values.len() != expected {
        return Err(ContractError::TensorDataMismatch);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ContractError::NonFiniteTensor);
    }
    if !spec.coordinate_scale.is_finite() || spec.coordinate_scale <= 0.0 {
        return Err(ContractError::UnsupportedInputContract);
    }
    if spec.coordinate_scale != 1.0 {
        let anchors = *spec
            .shape
            .last()
            .ok_or(ContractError::TensorShapeMismatch)?;
        if spec.shape.len() != 3 || spec.shape[1] < 4 {
            return Err(ContractError::TensorShapeMismatch);
        }
        for attribute in 0..4 {
            for value in &mut values[attribute * anchors..(attribute + 1) * anchors] {
                *value *= spec.coordinate_scale;
            }
        }
    }
    Ok(LogicalTensor {
        role: LogicalRole::Detections,
        shape: spec.shape.clone(),
        values,
    })
}

pub fn compare_raw_golden(
    reference: &LogicalTensor,
    candidate: &LogicalTensor,
    tolerance: &GoldenTolerance,
) -> Result<(), ContractError> {
    let absolute = tolerance
        .raw_absolute
        .ok_or(ContractError::MissingTolerance)?;
    let relative = tolerance
        .raw_relative
        .ok_or(ContractError::MissingTolerance)?;
    if !absolute.is_finite() || absolute < 0.0 || !relative.is_finite() || relative < 0.0 {
        return Err(ContractError::MissingTolerance);
    }
    if reference.role != LogicalRole::Detections
        || candidate.role != LogicalRole::Detections
        || reference.shape != candidate.shape
        || reference.values.len() != candidate.values.len()
    {
        return Err(ContractError::GoldenMismatch);
    }
    for (reference, candidate) in reference.values.iter().zip(&candidate.values) {
        if !reference.is_finite() || !candidate.is_finite() {
            return Err(ContractError::NonFiniteTensor);
        }
        if (reference - candidate).abs() > absolute + relative * reference.abs() {
            return Err(ContractError::GoldenMismatch);
        }
    }
    Ok(())
}

pub fn validate_golden_summary(
    summary: &GoldenSummary,
    tolerance: &GoldenTolerance,
) -> Result<(), ContractError> {
    if tolerance.raw_absolute.is_none()
        || tolerance.raw_relative.is_none()
        || tolerance.confidence_absolute.is_none()
        || tolerance.box_iou_minimum.is_none()
    {
        return Err(ContractError::MissingTolerance);
    }
    if summary.role != Some(LogicalRole::Detections)
        || summary.shape != [1, 84, 8400]
        || summary.element_count != 84 * 8400
        || summary.finite_count != summary.element_count
        || summary.raw_digest.len() != 64
        || !summary
            .raw_digest
            .bytes()
            .all(|value| value.is_ascii_hexdigit())
    {
        return Err(ContractError::GoldenMismatch);
    }
    Ok(())
}

pub fn resolve_postprocess_plan(spec: &TensorSpec) -> Result<PostprocessPlan, ContractError> {
    if spec.role != Some(LogicalRole::Detections)
        || spec.layout != TensorLayout::AttributesAnchors
        || spec.shape != [1, 84, 8400]
    {
        return Err(ContractError::UnsupportedInputContract);
    }
    Ok(PostprocessPlan {
        decode_in_operator: true,
        threshold_in_operator: true,
        nms_in_operator: !spec.nms_fused,
    })
}

fn ordered_rgb(image: &LogicalImage, layout: TensorLayout) -> Vec<u8> {
    match layout {
        TensorLayout::Nhwc => image.rgb.clone(),
        TensorLayout::Nchw => (0..3)
            .flat_map(|channel| {
                (0..image.width * image.height).map(move |pixel| image.rgb[pixel * 3 + channel])
            })
            .collect(),
        TensorLayout::AttributesAnchors => unreachable!(),
    }
}

fn product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
}

fn dequantize_u8(
    values: &[u8],
    quantization: Option<Quantization>,
) -> Result<Vec<f32>, ContractError> {
    let quantization = quantization.ok_or(ContractError::InvalidQuantization)?;
    if !quantization.scale.is_finite() || quantization.scale <= 0.0 {
        return Err(ContractError::InvalidQuantization);
    }
    Ok(values
        .iter()
        .map(|value| (*value as i32 - quantization.zero_point) as f32 * quantization.scale)
        .collect())
}

fn dequantize_i8(
    values: &[i8],
    quantization: Option<Quantization>,
) -> Result<Vec<f32>, ContractError> {
    let quantization = quantization.ok_or(ContractError::InvalidQuantization)?;
    if !quantization.scale.is_finite() || quantization.scale <= 0.0 {
        return Err(ContractError::InvalidQuantization);
    }
    Ok(values
        .iter()
        .map(|value| (*value as i32 - quantization.zero_point) as f32 * quantization.scale)
        .collect())
}
