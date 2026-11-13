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

/// Borrowed view of a single column value, independent of how the row is stored.
#[derive(Debug, Clone, Copy)]
pub enum Cell<'a> {
    Null,
    Int(i64),
    Str(&'a str),
    Vector(&'a [f32]),
}

impl Cell<'_> {
    pub fn to_json(&self) -> Result<JsonValue> {
        match self {
            Cell::Null => Ok(JsonValue::Null),
            Cell::Int(v) => Ok(JsonValue::Number((*v).into())),
            Cell::Str(s) => Ok(JsonValue::String((*s).to_string())),
            Cell::Vector(vec) => {
                let mut arr = Vec::with_capacity(vec.len());
                for &x in vec.iter() {
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

/// Column access shared by materialized rows and rows living in the columnar in-memory table.
pub trait RowAccess {
    fn cell(&self, idx: usize) -> Cell<'_>;
}

impl RowAccess for Row {
    #[inline]
    fn cell(&self, idx: usize) -> Cell<'_> {
        match &self.values[idx] {
            Value::Null => Cell::Null,
            Value::BigInt(v) => Cell::Int(*v),
            Value::Varchar(s) => Cell::Str(s.as_str()),
            Value::Vector(v) => Cell::Vector(v.as_slice()),
        }
    }
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
    if let Some(v) = parse_vector_fast(raw.as_bytes()) {
        return Ok(v);
    }
    parse_vector_str_slow(raw)
}

const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

#[inline]
fn is_plain_ws(b: u8) -> bool {
    b == b' ' || (0x09..=0x0d).contains(&b)
}

/// True when all eight bytes of `v` (little-endian load) are ASCII digits.
#[inline]
fn is_8_digits(v: u64) -> bool {
    ((v & 0xF0F0_F0F0_F0F0_F0F0) | (((v.wrapping_add(0x0606_0606_0606_0606)) & 0xF0F0_F0F0_F0F0_F0F0) >> 4))
        == 0x3333_3333_3333_3333
}

/// Convert eight ASCII digits (little-endian load, first digit in the low byte) to their value.
#[inline]
fn parse_8_digits(mut v: u64) -> u64 {
    const MASK: u64 = 0x0000_00FF_0000_00FF;
    const MUL1: u64 = 0x000F_4240_0000_0064; // 100 + (1_000_000 << 32)
    const MUL2: u64 = 0x0000_2710_0000_0001; // 1 + (10_000 << 32)
    v = v.wrapping_sub(0x3030_3030_3030_3030);
    v = v.wrapping_mul(10).wrapping_add(v >> 8);
    ((v & MASK).wrapping_mul(MUL1).wrapping_add(((v >> 16) & MASK).wrapping_mul(MUL2))) >> 32
}

/// Hand-written scanner for the common `[d.ddd, d.ddd, ...]` shape.
///
/// Values are produced exactly as `token.parse::<f64>() as f32` would:
/// * a plain decimal with at most 15 digits has an exactly representable integer
///   mantissa, and dividing it by an exact power of ten (<= 1e22) is one correctly
///   rounded IEEE operation, i.e. the correctly rounded double;
/// * with 16..=19 digits the quotient is within 1.5 ulp of the correctly rounded
///   double; it is used only when it is provably on the same side of every
///   single-precision rounding boundary, so the resulting `f32` is identical;
/// * every other token (exponent form, longer mantissa, explicit '+', ...) goes through
///   the standard library parser.
/// A literal this scanner cannot fully validate returns `None`, and the caller falls
/// back to the general implementation (which also produces the error messages).
fn parse_vector_fast(b: &[u8]) -> Option<Vec<f32>> {
    let mut start = 0;
    let mut end = b.len();
    while start < end && is_plain_ws(b[start]) {
        start += 1;
    }
    while end > start && is_plain_ws(b[end - 1]) {
        end -= 1;
    }
    if end - start < 2 || b[start] != b'[' || b[end - 1] != b']' {
        return None;
    }
    let s = &b[start + 1..end - 1];
    let n = s.len();
    let commas = s.iter().filter(|&&c| c == b',').count();
    let mut out: Vec<f32> = Vec::with_capacity(commas + 1);
    let mut i = 0usize;
    loop {
        while i < n && is_plain_ws(s[i]) {
            i += 1;
        }
        let tok_start = i;
        let neg = i < n && s[i] == b'-';
        if neg {
            i += 1;
        }
        let mut mant: u64 = 0;
        let int_start = i;
        while i < n && i - int_start < 19 {
            let d = s[i].wrapping_sub(b'0');
            if d > 9 {
                break;
            }
            mant = mant * 10 + d as u64;
            i += 1;
        }
        let mut digits = i - int_start;
        let mut frac = 0usize;
        if i < n && s[i] == b'.' {
            i += 1;
            let frac_start = i;
            while digits + (i - frac_start) + 8 <= 19 && i + 8 <= n {
                let mut raw = [0u8; 8];
                raw.copy_from_slice(&s[i..i + 8]);
                let w = u64::from_le_bytes(raw);
                if !is_8_digits(w) {
                    break;
                }
                mant = mant * 100_000_000 + parse_8_digits(w);
                i += 8;
            }
            while i < n && digits + (i - frac_start) < 19 {
                let d = s[i].wrapping_sub(b'0');
                if d > 9 {
                    break;
                }
                mant = mant * 10 + d as u64;
                i += 1;
            }
            frac = i - frac_start;
            digits += frac;
        }
        let simple_end = i;
        while i < n && is_plain_ws(s[i]) {
            i += 1;
        }
        let mut fast: Option<f32> = None;
        if (i == n || s[i] == b',') && digits >= 1 && frac <= 22 {
            let v = (mant as f64) / POW10[frac];
            let v = if neg { -v } else { v };
            if digits <= 15 || mant == 0 {
                fast = Some(v as f32);
            } else {
                // Distance (in double-precision ulps) from the nearest single-precision
                // rounding boundary; the quotient is at most 1.5 ulp off.
                let low = (v.to_bits() & 0x1FFF_FFFF) as i64;
                let a = v.abs();
                if (low - 0x1000_0000).abs() > 16 && a > 1e-30 && a < 1e30 {
                    fast = Some(v as f32);
                }
            }
        }
        let f = match fast {
            Some(f) => f,
            None => {
                // Irregular token: let the standard parser decide.
                let mut j = simple_end;
                while j < n && s[j] != b',' {
                    if s[j] >= 0x80 {
                        return None;
                    }
                    j += 1;
                }
                let mut tok_end = j;
                while tok_end > tok_start && is_plain_ws(s[tok_end - 1]) {
                    tok_end -= 1;
                }
                if tok_end == tok_start {
                    return None;
                }
                let tok = std::str::from_utf8(&s[tok_start..tok_end]).ok()?;
                i = j;
                let val = tok.parse::<f64>().ok()?;
                if !val.is_finite() {
                    return None;
                }
                val as f32
            }
        };
        if !f.is_finite() {
            return None;
        }
        out.push(f);
        if i == n {
            break;
        }
        // s[i] == b','
        i += 1;
    }
    Some(out)
}

fn parse_vector_str_slow(raw: &str) -> Result<Vec<f32>> {
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
