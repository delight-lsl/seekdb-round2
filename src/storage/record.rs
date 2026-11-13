use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::RowId;
use std::io::Read;

pub const DATA_FRAME_MAGIC: [u8; 4] = *b"MVD1";
pub const DATA_HEADER_LEN: usize = 17; // 4 (magic) + 1 (op) + 8 (row_id) + 4 (payload_len)
pub const DATA_TRAILER_LEN: usize = 4; // 4 (crc32)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataRecordOp {
    Put = 1,
    Tombstone = 2,
    /// Group terminator: every frame since the previous terminator is committed.
    Commit = 3,
    /// Group terminator of a WAL-coordinated transaction (`row_id` field carries the txn id).
    Pending = 4,
}

impl DataRecordOp {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            1 => Ok(Self::Put),
            2 => Ok(Self::Tombstone),
            3 => Ok(Self::Commit),
            4 => Ok(Self::Pending),
            _ => Err(DbError::new(
                ErrorCode::IoError,
                format!("invalid data record op {}", v),
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DataFrame {
    pub op: DataRecordOp,
    pub row_id: RowId,
    pub payload: Vec<u8>,
    pub frame_offset: u64,
    pub frame_len: u32,
}

/// Encode a single data frame into `out` and return its total byte length.
pub fn encode_data_frame(
    op: DataRecordOp,
    row_id: RowId,
    payload: &[u8],
    out: &mut Vec<u8>,
) -> Result<u32> {
    let start = out.len();
    let payload_len = u32::try_from(payload.len())
        .map_err(|_| DbError::new(ErrorCode::InternalError, "record payload exceeds u32"))?;
    out.extend_from_slice(&DATA_FRAME_MAGIC);
    out.push(op as u8);
    out.extend_from_slice(&row_id.to_le_bytes());
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(payload);
    let crc = crc32fast::hash(&out[start..]);
    out.extend_from_slice(&crc.to_le_bytes());
    let total = out.len() - start;
    u32::try_from(total)
        .map_err(|_| DbError::new(ErrorCode::InternalError, "frame len exceeds u32"))
}

/// Decode a frame from an exact byte buffer read via `pread`.
pub fn decode_frame_slice(buf: &[u8], frame_offset: u64) -> Result<DataFrame> {
    if buf.len() < DATA_HEADER_LEN + DATA_TRAILER_LEN {
        return Err(DbError::new(ErrorCode::IoError, "truncated data frame slice"));
    }
    if buf[0..4] != DATA_FRAME_MAGIC {
        return Err(DbError::new(ErrorCode::IoError, "bad data frame magic"));
    }
    let op = DataRecordOp::from_u8(buf[4])?;
    let row_id = u64::from_le_bytes(buf[5..13].try_into().unwrap());
    let payload_len = u32::from_le_bytes(buf[13..17].try_into().unwrap()) as usize;
    let expected_total = DATA_HEADER_LEN + payload_len + DATA_TRAILER_LEN;
    if buf.len() != expected_total {
        return Err(DbError::new(ErrorCode::IoError, "frame length mismatch"));
    }
    let expected_crc = u32::from_le_bytes(
        buf[DATA_HEADER_LEN + payload_len..expected_total]
            .try_into()
            .unwrap(),
    );
    let actual_crc = crc32fast::hash(&buf[..DATA_HEADER_LEN + payload_len]);
    if actual_crc != expected_crc {
        return Err(DbError::new(ErrorCode::IoError, "data frame CRC mismatch"));
    }
    Ok(DataFrame {
        op,
        row_id,
        payload: buf[DATA_HEADER_LEN..DATA_HEADER_LEN + payload_len].to_vec(),
        frame_offset,
        frame_len: expected_total as u32,
    })
}

/// Read next frame sequentially from a buffered reader at `offset`.
/// Returns `Ok(None)` on clean EOF, or `Err` on corrupted/incomplete frame.
pub fn read_next_data_frame<R: Read>(reader: &mut R, offset: u64) -> Result<Option<DataFrame>> {
    let mut header = [0u8; DATA_HEADER_LEN];
    let mut read_so_far = 0;
    while read_so_far < DATA_HEADER_LEN {
        let n = reader.read(&mut header[read_so_far..])?;
        if n == 0 {
            if read_so_far == 0 {
                return Ok(None);
            }
            return Err(DbError::new(
                ErrorCode::IoError,
                "truncated data frame header",
            ));
        }
        read_so_far += n;
    }
    if header[0..4] != DATA_FRAME_MAGIC {
        return Err(DbError::new(
            ErrorCode::IoError,
            "corrupted data frame magic",
        ));
    }
    let op = DataRecordOp::from_u8(header[4])?;
    let row_id = u64::from_le_bytes(header[5..13].try_into().unwrap());
    let payload_len = u32::from_le_bytes(header[13..17].try_into().unwrap()) as usize;

    let mut body = vec![0u8; payload_len + DATA_TRAILER_LEN];
    reader.read_exact(&mut body)?;

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&header);
    hasher.update(&body[..payload_len]);
    let actual_crc = hasher.finalize();
    let expected_crc = u32::from_le_bytes(body[payload_len..].try_into().unwrap());
    if actual_crc != expected_crc {
        return Err(DbError::new(
            ErrorCode::IoError,
            "corrupted data frame CRC32",
        ));
    }
    let total_len = (DATA_HEADER_LEN + payload_len + DATA_TRAILER_LEN) as u32;
    body.truncate(payload_len);
    Ok(Some(DataFrame {
        op,
        row_id,
        payload: body,
        frame_offset: offset,
        frame_len: total_len,
    }))
}

/// Begin a data frame in place inside `out`; returns the frame start offset.
/// The caller appends the payload directly and then calls `finish_data_frame`.
#[inline]
pub fn begin_data_frame(op: DataRecordOp, row_id: RowId, out: &mut Vec<u8>) -> usize {
    let start = out.len();
    out.extend_from_slice(&DATA_FRAME_MAGIC);
    out.push(op as u8);
    out.extend_from_slice(&row_id.to_le_bytes());
    out.extend_from_slice(&[0u8; 4]);
    start
}

/// Patch the payload length, append the CRC and return the total frame length.
#[inline]
pub fn finish_data_frame(start: usize, out: &mut Vec<u8>) -> Result<u32> {
    let payload_len = u32::try_from(out.len() - start - DATA_HEADER_LEN)
        .map_err(|_| DbError::new(ErrorCode::InternalError, "record payload exceeds u32"))?;
    out[start + 13..start + 17].copy_from_slice(&payload_len.to_le_bytes());
    let crc = crc32fast::hash(&out[start..]);
    out.extend_from_slice(&crc.to_le_bytes());
    u32::try_from(out.len() - start)
        .map_err(|_| DbError::new(ErrorCode::InternalError, "frame len exceeds u32"))
}
