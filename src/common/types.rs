use crate::common::error::{DbError, ErrorCode, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

pub type RowId = u64;
pub type TableId = u32;
pub type TxnId = u64;

/// Vector search execution mode controlled by protocol `search_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    #[default]
    Exact,
    Ann,
}

impl SearchMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "exact" => Ok(Self::Exact),
            "ann" => Ok(Self::Ann),
            other => Err(DbError::new(
                ErrorCode::InvalidArgument,
                format!("unsupported search_mode: {}", other),
            )),
        }
    }
}

/// Column data types supported by MiniVectorDB.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    BigInt,
    Varchar(usize),
    Vector(usize),
}

impl DataType {
    pub fn display_name(&self) -> String {
        match self {
            Self::BigInt => "BIGINT".to_string(),
            Self::Varchar(n) => format!("VARCHAR({})", n),
            Self::Vector(d) => format!("VECTOR({})", d),
        }
    }
}

/// Runtime column value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    BigInt(i64),
    Varchar(String),
    Vector(Vec<f32>),
}

impl Value {
    pub fn to_json(&self) -> Result<JsonValue> {
        match self {
            Self::Null => Ok(JsonValue::Null),
            Self::BigInt(v) => Ok(JsonValue::Number((*v).into())),
            Self::Varchar(s) => Ok(JsonValue::String(s.clone())),
            Self::Vector(vec) => {
                let mut arr = Vec::with_capacity(vec.len());
                for &x in vec {
                    let num = serde_json::Number::from_f64(x as f64).ok_or_else(|| {
                        DbError::new(ErrorCode::InvalidVector, "non-finite float in vector output")
                    })?;
                    arr.push(JsonValue::Number(num));
                }
                Ok(JsonValue::Array(arr))
            }
        }
    }
}

/// A full database row matching a table schema's column order.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub values: Vec<Value>,
}

/// Execution output for a single SQL statement.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecResult {
    Ok,
    AffectedRows(usize),
    Query {
        columns: Vec<String>,
        rows: Vec<Vec<JsonValue>>,
    },
}

/// Parse a string literal such as `"[1.0, 2.5, -3]"` into finite `f32` elements.
/// Rejects NaN, Infinity, float32 overflow, empty vectors, and malformed syntax.
pub fn parse_vector_str(raw: &str) -> Result<Vec<f32>> {
    let s = raw.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(DbError::new(
            ErrorCode::InvalidVector,
            format!("vector literal must be enclosed in [...]: {}", raw),
        ));
    }
    let inner = s[1..s.len() - 1].trim();
    if inner.is_empty() {
        return Err(DbError::new(
            ErrorCode::InvalidVector,
            "vector literal cannot be empty",
        ));
    }
    let est_dim = inner.bytes().filter(|&b| b == b',').count() + 1;
    let mut result = Vec::with_capacity(est_dim);
    for part in inner.split(',') {
        let token = part.trim();
        if token.is_empty() {
            return Err(DbError::new(
                ErrorCode::InvalidVector,
                format!("empty element in vector literal: {}", raw),
            ));
        }
        let val_f64: f64 = token.parse().map_err(|_| {
            DbError::new(
                ErrorCode::InvalidVector,
                format!("invalid float element '{}' in vector", token),
            )
        })?;
        if !val_f64.is_finite() {
            return Err(DbError::new(
                ErrorCode::InvalidVector,
                format!("non-finite float '{}' in vector", token),
            ));
        }
        let val_f32 = val_f64 as f32;
        if !val_f32.is_finite() {
            return Err(DbError::new(
                ErrorCode::InvalidVector,
                format!("float32 overflow for '{}' in vector", token),
            ));
        }
        result.push(val_f32);
    }
    Ok(result)
}

/// Convert raw f64 slice (from bracket syntax) to finite `Vec<f32>`.
pub fn validate_f64_slice_as_f32(vals: &[f64]) -> Result<Vec<f32>> {
    if vals.is_empty() {
        return Err(DbError::new(
            ErrorCode::InvalidVector,
            "vector literal cannot be empty",
        ));
    }
    let mut out = Vec::with_capacity(vals.len());
    for &v in vals {
        if !v.is_finite() {
            return Err(DbError::new(
                ErrorCode::InvalidVector,
                "non-finite float in vector",
            ));
        }
        let f = v as f32;
        if !f.is_finite() {
            return Err(DbError::new(
                ErrorCode::InvalidVector,
                "float32 overflow in vector",
            ));
        }
        out.push(f);
    }
    Ok(out)
}
