use crate::catalog::schema::VectorIndexSchema;
use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::TxnId;
use crate::storage::codec::{
    decode_schema, encode_schema, write_i64, write_str, write_u16, write_u32, write_u64, write_u8,
    Cursor,
};
use crate::storage::file::sync_dir;
use crate::transaction::operation::WalOp;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const WAL_MAGIC: [u8; 4] = *b"WAL1";
pub const WAL_HEADER_LEN: usize = 17; // 4 (magic) + 8 (txn_id) + 1 (kind) + 4 (payload_len)
pub const WAL_TRAILER_LEN: usize = 4; // 4 (crc32)
const WAL_CHECKPOINT_THRESHOLD: u64 = 16 * 1024 * 1024; // 16 MiB

pub const KIND_BEGIN: u8 = 1;
pub const KIND_CREATE_TABLE: u8 = 2;
pub const KIND_CREATE_INDEX: u8 = 3;
pub const KIND_PUT_ROW: u8 = 4;
pub const KIND_DELETE_ROW: u8 = 5;
pub const KIND_COMMIT: u8 = 6;

#[derive(Debug, Clone)]
pub enum WalFrameKind {
    Begin,
    Op(WalOp),
    Commit,
}

#[derive(Debug, Clone)]
pub struct WalFrame {
    pub txn_id: TxnId,
    pub kind: WalFrameKind,
}

#[derive(Debug)]
pub struct WalScanResult {
    pub frames: Vec<WalFrame>,
    pub valid_end_offset: u64,
    pub truncated_tail: bool,
}

pub fn wal_segment_path(wal_dir: &Path, seq: u64) -> PathBuf {
    wal_dir.join(format!("wal_{:08}.log", seq))
}

pub fn list_wal_segments(wal_dir: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let mut segs = Vec::new();
    if !wal_dir.exists() {
        return Ok(segs);
    }
    for entry in fs::read_dir(wal_dir)? {
        let entry = entry?;
        if let Some(name) = entry.file_name().to_str() {
            if let Some(rest) = name.strip_prefix("wal_") {
                if let Some(num_str) = rest.strip_suffix(".log") {
                    if let Ok(seq) = num_str.parse::<u64>() {
                        segs.push((seq, entry.path()));
                    }
                }
            }
        }
    }
    segs.sort_by_key(|(seq, _)| *seq);
    Ok(segs)
}

pub fn encode_wal_frame(txn_id: TxnId, kind: u8, payload: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let start = out.len();
    let payload_len = u32::try_from(payload.len())
        .map_err(|_| DbError::new(ErrorCode::InternalError, "WAL payload exceeds u32"))?;
    out.extend_from_slice(&WAL_MAGIC);
    out.extend_from_slice(&txn_id.to_le_bytes());
    out.push(kind);
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(payload);
    let crc = crc32fast::hash(&out[start..]);
    out.extend_from_slice(&crc.to_le_bytes());
    Ok(())
}

pub fn encode_wal_op(op: &WalOp, buf: &mut Vec<u8>) -> Result<(u8, ())> {
    match op {
        WalOp::CreateTable(schema) => {
            encode_schema(schema, buf)?;
            Ok((KIND_CREATE_TABLE, ()))
        }
        WalOp::CreateIndex { table_id, index } => {
            write_u32(buf, *table_id);
            write_str(buf, &index.name)?;
            write_str(buf, &index.column_name)?;
            write_u16(buf, index.col_idx as u16);
            write_u32(buf, index.dim as u32);
            Ok((KIND_CREATE_INDEX, ()))
        }
        WalOp::PutRow {
            table_id,
            row_id,
            old_pk,
            row_bytes,
        } => {
            write_u32(buf, *table_id);
            write_u64(buf, *row_id);
            match old_pk {
                Some(pk) => {
                    write_u8(buf, 1);
                    write_i64(buf, *pk);
                }
                None => {
                    write_u8(buf, 0);
                    write_i64(buf, 0);
                }
            }
            write_u32(buf, row_bytes.len() as u32);
            buf.extend_from_slice(row_bytes);
            Ok((KIND_PUT_ROW, ()))
        }
        WalOp::DeleteRow {
            table_id,
            row_id,
            old_pk,
        } => {
            write_u32(buf, *table_id);
            write_u64(buf, *row_id);
            match old_pk {
                Some(pk) => {
                    write_u8(buf, 1);
                    write_i64(buf, *pk);
                }
                None => {
                    write_u8(buf, 0);
                    write_i64(buf, 0);
                }
            }
            Ok((KIND_DELETE_ROW, ()))
        }
    }
}

fn decode_wal_op(kind: u8, payload: &[u8]) -> Result<WalFrameKind> {
    let mut cur = Cursor::new(payload);
    match kind {
        KIND_BEGIN => {
            if cur.remaining() != 0 {
                return Err(DbError::new(ErrorCode::IoError, "extra bytes in WAL BEGIN"));
            }
            Ok(WalFrameKind::Begin)
        }
        KIND_COMMIT => {
            if cur.remaining() != 0 {
                return Err(DbError::new(ErrorCode::IoError, "extra bytes in WAL COMMIT"));
            }
            Ok(WalFrameKind::Commit)
        }
        KIND_CREATE_TABLE => {
            let schema = decode_schema(&mut cur)?;
            if cur.remaining() != 0 {
                return Err(DbError::new(ErrorCode::IoError, "trailing bytes in WAL CreateTable"));
            }
            Ok(WalFrameKind::Op(WalOp::CreateTable(schema)))
        }
        KIND_CREATE_INDEX => {
            let table_id = cur.read_u32()?;
            let name = cur.read_str()?;
            let column_name = cur.read_str()?;
            let col_idx = cur.read_u16()? as usize;
            let dim = cur.read_u32()? as usize;
            if cur.remaining() != 0 {
                return Err(DbError::new(ErrorCode::IoError, "trailing bytes in WAL CreateIndex"));
            }
            Ok(WalFrameKind::Op(WalOp::CreateIndex {
                table_id,
                index: VectorIndexSchema {
                    name,
                    column_name,
                    col_idx,
                    dim,
                },
            }))
        }
        KIND_PUT_ROW => {
            let table_id = cur.read_u32()?;
            let row_id = cur.read_u64()?;
            let has_pk = cur.read_u8()? != 0;
            let pk_raw = cur.read_i64()?;
            let old_pk = if has_pk { Some(pk_raw) } else { None };
            let rlen = cur.read_u32()? as usize;
            let row_bytes = cur.read_bytes(rlen)?.to_vec();
            if cur.remaining() != 0 {
                return Err(DbError::new(ErrorCode::IoError, "trailing bytes in WAL PutRow"));
            }
            Ok(WalFrameKind::Op(WalOp::PutRow {
                table_id,
                row_id,
                old_pk,
                row_bytes,
            }))
        }
        KIND_DELETE_ROW => {
            let table_id = cur.read_u32()?;
            let row_id = cur.read_u64()?;
            let has_pk = cur.read_u8()? != 0;
            let pk_raw = cur.read_i64()?;
            let old_pk = if has_pk { Some(pk_raw) } else { None };
            if cur.remaining() != 0 {
                return Err(DbError::new(ErrorCode::IoError, "trailing bytes in WAL DeleteRow"));
            }
            Ok(WalFrameKind::Op(WalOp::DeleteRow {
                table_id,
                row_id,
                old_pk,
            }))
        }
        _ => Err(DbError::new(
            ErrorCode::IoError,
            format!("unknown WAL frame kind {}", kind),
        )),
    }
}

/// Scan a WAL segment starting at `start_offset`.
/// Any incomplete or CRC-failing tail frame cleanly stops scanning and reports `valid_end_offset`.
pub fn scan_wal_segment(path: &Path, start_offset: u64) -> Result<WalScanResult> {
    let file = OpenOptions::new().read(true).write(true).open(path)?;
    file.sync_all()?;
    let file_len = file.metadata()?.len();
    if start_offset >= file_len {
        return Ok(WalScanResult {
            frames: Vec::new(),
            valid_end_offset: file_len,
            truncated_tail: false,
        });
    }

    let mut reader_file = file.try_clone()?;
    reader_file.seek(SeekFrom::Start(start_offset))?;
    let mut reader = BufReader::with_capacity(256 * 1024, reader_file);

    let mut frames = Vec::new();
    let mut offset = start_offset;
    let mut truncated_tail = false;

    loop {
        let mut header = [0u8; WAL_HEADER_LEN];
        let mut got = 0;
        let mut hit_eof = false;
        while got < WAL_HEADER_LEN {
            let n = reader.read(&mut header[got..])?;
            if n == 0 {
                hit_eof = true;
                break;
            }
            got += n;
        }
        if hit_eof {
            if got > 0 {
                truncated_tail = true;
            }
            break;
        }
        if header[0..4] != WAL_MAGIC {
            truncated_tail = true;
            break;
        }
        let txn_id = u64::from_le_bytes(header[4..12].try_into().unwrap());
        let kind = header[12];
        let payload_len = u32::from_le_bytes(header[13..17].try_into().unwrap()) as usize;

        // Sanity bound to avoid OOM on corrupt length field
        if offset + (WAL_HEADER_LEN as u64) + (payload_len as u64) + (WAL_TRAILER_LEN as u64)
            > file_len
        {
            truncated_tail = true;
            break;
        }

        let mut body = vec![0u8; payload_len + WAL_TRAILER_LEN];
        if reader.read_exact(&mut body).is_err() {
            truncated_tail = true;
            break;
        }

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&header);
        hasher.update(&body[..payload_len]);
        let actual_crc = hasher.finalize();
        let expected_crc = u32::from_le_bytes(body[payload_len..].try_into().unwrap());
        if actual_crc != expected_crc {
            truncated_tail = true;
            break;
        }

        let Ok(decoded_kind) = decode_wal_op(kind, &body[..payload_len]) else {
            truncated_tail = true;
            break;
        };

        frames.push(WalFrame {
            txn_id,
            kind: decoded_kind,
        });
        offset += (WAL_HEADER_LEN + payload_len + WAL_TRAILER_LEN) as u64;
    }

    if truncated_tail {
        file.set_len(offset)?;
        file.sync_all()?;
    }

    Ok(WalScanResult {
        frames,
        valid_end_offset: offset,
        truncated_tail,
    })
}

/// Write-Ahead Log manager handling durable commit records and segment rotation.
#[derive(Debug)]
pub struct WalManager {
    pub wal_dir: PathBuf,
    pub current_seq: u64,
    pub current_offset: u64,
    pub next_txn_id: TxnId,
    file: File,
}

impl WalManager {
    pub fn open(wal_dir: PathBuf, seq: u64, offset: u64, next_txn_id: TxnId) -> Result<Self> {
        fs::create_dir_all(&wal_dir)?;
        let path = wal_segment_path(&wal_dir, seq);
        let is_new = !path.exists();
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;
        if is_new {
            sync_dir(&wal_dir)?;
        }
        let actual_len = file.metadata()?.len();
        let cur_off = offset.min(actual_len);
        file.seek(SeekFrom::Start(cur_off))?;
        Ok(Self {
            wal_dir,
            current_seq: seq,
            current_offset: cur_off,
            next_txn_id: next_txn_id.max(1),
            file,
        })
    }

    /// Write all operations of a transaction plus COMMIT frame into the WAL and `fsync`.
    pub fn append_committed_txn(&mut self, ops: &[WalOp]) -> Result<TxnId> {
        if ops.is_empty() {
            return Ok(0);
        }
        let txn_id = self.next_txn_id;
        self.next_txn_id += 1;

        let mut buf = Vec::with_capacity(256);
        encode_wal_frame(txn_id, KIND_BEGIN, &[], &mut buf)?;
        let mut op_payload = Vec::new();
        for op in ops {
            op_payload.clear();
            let (kind, ()) = encode_wal_op(op, &mut op_payload)?;
            encode_wal_frame(txn_id, kind, &op_payload, &mut buf)?;
        }
        encode_wal_frame(txn_id, KIND_COMMIT, &[], &mut buf)?;

        self.file.seek(SeekFrom::Start(self.current_offset))?;
        self.file.write_all(&buf)?;
        self.file.flush()?;
        self.file.sync_all()?;
        self.current_offset += buf.len() as u64;
        Ok(txn_id)
    }

    pub fn should_checkpoint(&self) -> bool {
        self.current_offset >= WAL_CHECKPOINT_THRESHOLD
    }

    pub fn rotate_segment(&mut self) -> Result<(u64, u64)> {
        self.file.sync_all()?;
        let next_seq = self.current_seq + 1;
        let path = wal_segment_path(&self.wal_dir, next_seq);
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        file.sync_all()?;
        sync_dir(&self.wal_dir)?;
        self.file = file;
        self.current_seq = next_seq;
        self.current_offset = 0;
        Ok((next_seq, 0))
    }

    pub fn delete_segments_before(&self, min_seq: u64) -> Result<()> {
        let mut deleted_any = false;
        for (seq, path) in list_wal_segments(&self.wal_dir)? {
            if seq < min_seq {
                let _ = fs::remove_file(path);
                deleted_any = true;
            }
        }
        if deleted_any {
            sync_dir(&self.wal_dir)?;
        }
        Ok(())
    }
}
