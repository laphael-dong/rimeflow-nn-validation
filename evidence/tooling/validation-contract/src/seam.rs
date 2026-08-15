//! 仅供第 2 阶段红测使用的确定性 seam。
//!
//! 第 5 阶段接入真实 manifest/adapter 时，应以生产实现替换这些入口；本阶段不得在这里
//! 预先实现 layout、dtype、量化、逻辑 role 或后处理责任分派。

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
    NotImplemented(&'static str),
    MissingLogicalRole,
    UnsupportedInputContract,
    InvalidQuantization,
    GoldenMismatch,
    MissingTolerance,
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotImplemented(feature) => write!(formatter, "not_implemented({feature})"),
            Self::MissingLogicalRole => formatter.write_str("missing_logical_role"),
            Self::UnsupportedInputContract => formatter.write_str("unsupported_input_contract"),
            Self::InvalidQuantization => formatter.write_str("invalid_quantization"),
            Self::GoldenMismatch => formatter.write_str("golden_mismatch"),
            Self::MissingTolerance => formatter.write_str("missing_tolerance"),
        }
    }
}

pub fn prepare_input(
    _image: &LogicalImage,
    _spec: &TensorSpec,
) -> Result<RuntimeTensor, ContractError> {
    Err(ContractError::NotImplemented(
        "logical image layout/dtype/quantization adapter",
    ))
}

pub fn normalize_output(
    _spec: &TensorSpec,
    _runtime_tensor: &RuntimeTensor,
) -> Result<LogicalTensor, ContractError> {
    Err(ContractError::NotImplemented(
        "runtime output to logical role mapping",
    ))
}

pub fn compare_raw_golden(
    _reference: &LogicalTensor,
    _candidate: &LogicalTensor,
    _tolerance: &GoldenTolerance,
) -> Result<(), ContractError> {
    Err(ContractError::NotImplemented(
        "frozen raw/decoded golden comparison",
    ))
}

pub fn validate_golden_summary(
    _summary: &GoldenSummary,
    _tolerance: &GoldenTolerance,
) -> Result<(), ContractError> {
    Err(ContractError::NotImplemented(
        "logical role and golden summary validation",
    ))
}

pub fn resolve_postprocess_plan(_spec: &TensorSpec) -> Result<PostprocessPlan, ContractError> {
    Err(ContractError::NotImplemented(
        "single postprocess responsibility plan",
    ))
}
