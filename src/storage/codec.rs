use crate::catalog::schema::{ColumnSchema, TableSchema, VectorIndexSchema};
use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::{DataType, Row, Value};

#[inline]
pub fn write_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}

#[inline]
pub fn write_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn write_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn write_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_str(buf: &mut Vec<u8>, s: &str) -> Result<()> {
    let bytes = s.as_bytes();
    let len = u32::try_from(bytes.len())
        .map_err(|_| DbError::new(ErrorCode::ValueTooLong, "string exceeds u32 byte length"))?;
    write_u32(buf, len);
    buf.extend_from_slice(bytes);
    Ok(())
}

pub struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    pub fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        if self.remaining() < len {
            return Err(DbError::new(
                ErrorCode::IoError,
                "unexpected EOF while decoding binary record",
            ));
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    pub fn skip(&mut self, len: usize) -> Result<()> {
        if self.remaining() < len {
            return Err(DbError::new(
                ErrorCode::IoError,
                "unexpected EOF while skipping binary bytes",
            ));
        }
        self.pos += len;
        Ok(())
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    pub fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn read_i64(&mut self) -> Result<i64> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn read_f32(&mut self) -> Result<f32> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn read_str(&mut self) -> Result<String> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_string())
            .map_err(|_| DbError::new(ErrorCode::IoError, "invalid UTF-8 in stored string"))
    }
}

const TAG_NULL: u8 = 0;
const TAG_BIGINT: u8 = 1;
const TAG_VARCHAR: u8 = 2;
const TAG_VECTOR: u8 = 3;

/// Encode a `Row` into explicit little-endian bytes.
pub fn encode_row(row: &Row, buf: &mut Vec<u8>) -> Result<()> {
    let col_count = u16::try_from(row.values.len())
        .map_err(|_| DbError::new(ErrorCode::InternalError, "too many columns in row"))?;
    write_u16(buf, col_count);
    for val in &row.values {
        match val {
            Value::Null => write_u8(buf, TAG_NULL),
            Value::BigInt(v) => {
                write_u8(buf, TAG_BIGINT);
                write_i64(buf, *v);
            }
            Value::Varchar(s) => {
                write_u8(buf, TAG_VARCHAR);
                write_str(buf, s)?;
            }
            Value::Vector(vec) => {
                write_u8(buf, TAG_VECTOR);
                let dim = u32::try_from(vec.len()).map_err(|_| {
                    DbError::new(ErrorCode::InvalidVector, "vector dimension exceeds u32")
                })?;
                write_u32(buf, dim);
                for &x in vec {
                    write_f32(buf, x);
                }
            }
        }
    }
    Ok(())
}

/// Decode all columns of a `Row`.
pub fn decode_row(bytes: &[u8], schema: &TableSchema) -> Result<Row> {
    let mut cur = Cursor::new(bytes);
    let col_count = cur.read_u16()? as usize;
    if col_count != schema.columns.len() {
        return Err(DbError::new(
            ErrorCode::IoError,
            format!(
                "stored row column count {} != schema {}",
                col_count,
                schema.columns.len()
            ),
        ));
    }
    let mut values = Vec::with_capacity(col_count);
    for _ in 0..col_count {
        values.push(read_value(&mut cur)?);
    }
    Ok(Row { values })
}

/// Decode only needed columns (sets unrequested columns to `Value::Null` without allocation).
pub fn decode_row_selective(bytes: &[u8], schema: &TableSchema, needed: &[bool]) -> Result<Row> {
    let mut cur = Cursor::new(bytes);
    let col_count = cur.read_u16()? as usize;
    if col_count != schema.columns.len() || needed.len() != col_count {
        return Err(DbError::new(ErrorCode::IoError, "column count mismatch in selective decode"));
    }
    let mut values = Vec::with_capacity(col_count);
    for &need in needed {
        if need {
            values.push(read_value(&mut cur)?);
        } else {
            skip_value(&mut cur)?;
            values.push(Value::Null);
        }
    }
    Ok(Row { values })
}

/// Extract primary key value quickly if present, skipping all other columns.
pub fn extract_pk_from_row_bytes(bytes: &[u8], pk_idx: usize) -> Result<Option<i64>> {
    let mut cur = Cursor::new(bytes);
    let col_count = cur.read_u16()? as usize;
    if pk_idx >= col_count {
        return Err(DbError::new(ErrorCode::IoError, "pk_idx out of bounds"));
    }
    for i in 0..col_count {
        if i == pk_idx {
            let tag = cur.read_u8()?;
            return match tag {
                TAG_NULL => Ok(None),
                TAG_BIGINT => Ok(Some(cur.read_i64()?)),
                _ => Err(DbError::new(ErrorCode::IoError, "non-bigint PK in record")),
            };
        } else {
            skip_value(&mut cur)?;
        }
    }
    Ok(None)
}

/// Extract a single vector column directly from encoded row bytes without decoding other columns.
pub fn extract_vector_from_row_bytes(bytes: &[u8], col_idx: usize) -> Result<Option<Vec<f32>>> {
    let mut cur = Cursor::new(bytes);
    let col_count = cur.read_u16()? as usize;
    if col_idx >= col_count {
        return Err(DbError::new(ErrorCode::IoError, "vector col_idx out of bounds"));
    }
    for _ in 0..col_idx {
        skip_value(&mut cur)?;
    }
    match read_value(&mut cur)? {
        Value::Null => Ok(None),
        Value::Vector(v) => Ok(Some(v)),
        _ => Err(DbError::new(ErrorCode::IoError, "expected vector value in record")),
    }
}

fn read_value(cur: &mut Cursor<'_>) -> Result<Value> {
    let tag = cur.read_u8()?;
    match tag {
        TAG_NULL => Ok(Value::Null),
        TAG_BIGINT => Ok(Value::BigInt(cur.read_i64()?)),
        TAG_VARCHAR => Ok(Value::Varchar(cur.read_str()?)),
        TAG_VECTOR => {
            let dim = cur.read_u32()? as usize;
            let raw = cur.read_bytes(dim * 4)?;
            let mut vec = Vec::with_capacity(dim);
            for chunk in raw.chunks_exact(4) {
                let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                vec.push(f);
            }
            Ok(Value::Vector(vec))
        }
        other => Err(DbError::new(
            ErrorCode::IoError,
            format!("unknown encoded value tag {}", other),
        )),
    }
}

fn skip_value(cur: &mut Cursor<'_>) -> Result<()> {
    let tag = cur.read_u8()?;
    match tag {
        TAG_NULL => Ok(()),
        TAG_BIGINT => cur.skip(8),
        TAG_VARCHAR => {
            let len = cur.read_u32()? as usize;
            cur.skip(len)
        }
        TAG_VECTOR => {
            let dim = cur.read_u32()? as usize;
            cur.skip(dim * 4)
        }
        other => Err(DbError::new(
            ErrorCode::IoError,
            format!("unknown encoded value tag {}", other),
        )),
    }
}

pub fn encode_schema(schema: &TableSchema, buf: &mut Vec<u8>) -> Result<()> {
    write_u32(buf, schema.table_id);
    write_str(buf, &schema.name)?;
    let col_len = u16::try_from(schema.columns.len())
        .map_err(|_| DbError::new(ErrorCode::InternalError, "too many columns"))?;
    write_u16(buf, col_len);
    for col in &schema.columns {
        write_str(buf, &col.name)?;
        match col.data_type {
            DataType::BigInt => {
                write_u8(buf, 1);
                write_u32(buf, 0);
            }
            DataType::Varchar(n) => {
                write_u8(buf, 2);
                write_u32(buf, n as u32);
            }
            DataType::Vector(d) => {
                write_u8(buf, 3);
                write_u32(buf, d as u32);
            }
        }
        write_u8(buf, if col.is_primary_key { 1 } else { 0 });
    }
    match schema.pk_col_idx {
        Some(idx) => {
            write_u8(buf, 1);
            write_u16(buf, idx as u16);
        }
        None => {
            write_u8(buf, 0);
            write_u16(buf, 0);
        }
    }
    let idx_len = u16::try_from(schema.indexes.len())
        .map_err(|_| DbError::new(ErrorCode::InternalError, "too many indexes"))?;
    write_u16(buf, idx_len);
    for idx in &schema.indexes {
        write_str(buf, &idx.name)?;
        write_str(buf, &idx.column_name)?;
        write_u16(buf, idx.col_idx as u16);
        write_u32(buf, idx.dim as u32);
    }
    Ok(())
}

pub fn decode_schema(cur: &mut Cursor<'_>) -> Result<TableSchema> {
    let table_id = cur.read_u32()?;
    let name = cur.read_str()?;
    let col_len = cur.read_u16()? as usize;
    let mut columns = Vec::with_capacity(col_len);
    for _ in 0..col_len {
        let cname = cur.read_str()?;
        let ttag = cur.read_u8()?;
        let targ = cur.read_u32()? as usize;
        let data_type = match ttag {
            1 => DataType::BigInt,
            2 => DataType::Varchar(targ),
            3 => DataType::Vector(targ),
            _ => {
                return Err(DbError::new(
                    ErrorCode::IoError,
                    format!("invalid column type tag {}", ttag),
                ));
            }
        };
        let is_primary_key = cur.read_u8()? != 0;
        columns.push(ColumnSchema {
            name: cname,
            data_type,
            is_primary_key,
        });
    }
    let has_pk = cur.read_u8()? != 0;
    let pk_idx_raw = cur.read_u16()? as usize;
    let pk_col_idx = if has_pk { Some(pk_idx_raw) } else { None };
    let idx_len = cur.read_u16()? as usize;
    let mut indexes = Vec::with_capacity(idx_len);
    for _ in 0..idx_len {
        let iname = cur.read_str()?;
        let col_name = cur.read_str()?;
        let col_idx = cur.read_u16()? as usize;
        let dim = cur.read_u32()? as usize;
        indexes.push(VectorIndexSchema {
            name: iname,
            column_name: col_name,
            col_idx,
            dim,
        });
    }
    Ok(TableSchema {
        table_id,
        name,
        columns,
        pk_col_idx,
        indexes,
    })
}
