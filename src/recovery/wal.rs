use crate::catalog::schema::VectorIndexSchema;
use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::TxnId;
use crate::storage::codec::{decode_schema, encode_schema, write_str, write_u16, write_u32, Cursor};
use crate::storage::file::{commit_sync, sync_dir};
use crate::transaction::operation::WalOp;
use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

pub const WAL_MAGIC: [u8; 4] = *b"WAL2";
pub const WAL_HEADER_LEN: usize = 17; // 4 (magic) + 8 (txn_id) + 1 (kind) + 4 (payload_len)
pub const WAL_TRAILER_LEN: usize = 4; // 4 (crc32)
const WAL_CHECKPOINT_THRESHOLD: u64 = 4 * 1024 * 1024;

pub const KIND_BEGIN: u8 = 1;
pub const KIND_CREATE_TABLE: u8 = 2;
pub const KIND_CREATE_INDEX: u8 = 3;
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

pub fn encode_wal_op(op: &WalOp, buf: &mut Vec<u8>) -> Result<u8> {
    match op {
        WalOp::CreateTable(schema) => {
            encode_schema(schema, buf)?;
            Ok(KIND_CREATE_TABLE)
        }
        WalOp::CreateIndex { table_id, index } => {
            write_u32(buf, *table_id);
            write_str(buf, &index.name)?;
            write_str(buf, &index.column_name)?;
            write_u16(buf, index.col_idx as u16);
            write_u32(buf, index.dim as u32);
            Ok(KIND_CREATE_INDEX)
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
        _ => Err(DbError::new(
            ErrorCode::IoError,
            format!("unknown WAL frame kind {}", kind),
        )),
    }
}

/// Scan a WAL segment starting at `start_offset`.
/// Any incomplete or CRC-failing tail frame cleanly stops scanning; the torn tail is cut off.
pub fn scan_wal_segment(path: &Path, start_offset: u64) -> Result<WalScanResult> {
    let file = OpenOptions::new().read(true).write(true).open(path)?;
    let file_len = file.metadata()?.len();
    if start_offset >= file_len {
        return Ok(WalScanResult {
            frames: Vec::new(),
            valid_end_offset: file_len,
            truncated_tail: false,
        });
    }
    let mut bytes = vec![0u8; (file_len - start_offset) as usize];
    file.read_exact_at(&mut bytes, start_offset)?;

    let mut frames = Vec::new();
    let mut pos = 0usize;
    let mut truncated_tail = false;
    while pos < bytes.len() {
        if bytes.len() - pos < WAL_HEADER_LEN + WAL_TRAILER_LEN {
            truncated_tail = true;
            break;
        }
        let header = &bytes[pos..pos + WAL_HEADER_LEN];
        if header[0..4] != WAL_MAGIC {
            truncated_tail = true;
            break;
        }
        let txn_id = u64::from_le_bytes(header[4..12].try_into().unwrap());
        let kind = header[12];
        let payload_len = u32::from_le_bytes(header[13..17].try_into().unwrap()) as usize;
        let total = WAL_HEADER_LEN + payload_len + WAL_TRAILER_LEN;
        if bytes.len() - pos < total {
            truncated_tail = true;
            break;
        }
        let body_end = pos + WAL_HEADER_LEN + payload_len;
        let expected_crc = u32::from_le_bytes(bytes[body_end..body_end + 4].try_into().unwrap());
        if crc32fast::hash(&bytes[pos..body_end]) != expected_crc {
            truncated_tail = true;
            break;
        }
        let Ok(decoded_kind) = decode_wal_op(kind, &bytes[pos + WAL_HEADER_LEN..body_end]) else {
            truncated_tail = true;
            break;
        };
        frames.push(WalFrame {
            txn_id,
            kind: decoded_kind,
        });
        pos += total;
    }

    let valid_end_offset = start_offset + pos as u64;
    if truncated_tail {
        file.set_len(valid_end_offset)?;
        file.sync_all()?;
    }
    Ok(WalScanResult {
        frames,
        valid_end_offset,
        truncated_tail,
    })
}

/// Write-ahead log for schema changes and for the commit point of transactions that
/// span more than one table file. Row data never goes through this log: it is written
/// once, straight into the table files.
#[derive(Debug)]
pub struct WalManager {
    pub wal_dir: PathBuf,
    pub current_seq: u64,
    pub current_offset: u64,
    pub next_txn_id: TxnId,
    pub dirty_since_checkpoint: bool,
    file: File,
}

impl WalManager {
    pub fn open(wal_dir: PathBuf, seq: u64, offset: u64, next_txn_id: TxnId) -> Result<Self> {
        fs::create_dir_all(&wal_dir)?;
        let path = wal_segment_path(&wal_dir, seq);
        let is_new = !path.exists();
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;
        if is_new {
            sync_dir(&wal_dir)?;
        }
        let actual_len = file.metadata()?.len();
        let cur_off = offset.min(actual_len);
        Ok(Self {
            wal_dir,
            current_seq: seq,
            current_offset: cur_off,
            next_txn_id: next_txn_id.max(1),
            dirty_since_checkpoint: cur_off > 0,
            file,
        })
    }

    pub fn alloc_txn_id(&mut self) -> TxnId {
        let id = self.next_txn_id;
        self.next_txn_id += 1;
        id
    }

    /// Durably record transaction `txn_id` as committed, together with its schema operations.
    pub fn append_commit(&mut self, txn_id: TxnId, ops: &[WalOp]) -> Result<()> {
        let mut buf = Vec::with_capacity(256);
        encode_wal_frame(txn_id, KIND_BEGIN, &[], &mut buf)?;
        let mut op_payload = Vec::new();
        for op in ops {
            op_payload.clear();
            let kind = encode_wal_op(op, &mut op_payload)?;
            encode_wal_frame(txn_id, kind, &op_payload, &mut buf)?;
        }
        encode_wal_frame(txn_id, KIND_COMMIT, &[], &mut buf)?;

        let res = self
            .file
            .write_all_at(&buf, self.current_offset)
            .and_then(|_| commit_sync(&self.file, self.current_offset, buf.len() as u64));
        if let Err(e) = res {
            let _ = self.file.set_len(self.current_offset);
            return Err(e.into());
        }
        self.current_offset += buf.len() as u64;
        self.dirty_since_checkpoint = true;
        Ok(())
    }

    /// Allocate a transaction id and durably commit `ops` under it.
    pub fn append_committed_txn(&mut self, ops: &[WalOp]) -> Result<TxnId> {
        let txn_id = self.alloc_txn_id();
        self.append_commit(txn_id, ops)?;
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
        self.dirty_since_checkpoint = false;
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
