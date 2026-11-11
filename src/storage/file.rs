use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::RowId;
use crate::storage::codec::encode_row;
use crate::storage::record::{
    decode_frame_slice, encode_data_frame, read_next_data_frame, DataFrame, DataRecordOp,
};
use crate::common::types::Row;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// Fsync a directory path so file creations/renames/deletions are durable on Linux.
pub fn sync_dir(dir: &Path) -> Result<()> {
    let f = File::open(dir)?;
    f.sync_all()?;
    Ok(())
}

/// Atomically write bytes to `target` using `tmp_path` -> `sync_all` -> `rename` -> `sync_dir`.
pub fn atomic_write_file(target: &Path, tmp_path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = tmp_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(tmp_path)?;
        f.write_all(bytes)?;
        f.flush()?;
        f.sync_all()?;
    }
    fs::rename(tmp_path, target)?;
    if let Some(parent) = target.parent() {
        sync_dir(parent)?;
    }
    Ok(())
}

/// Append-only table heap file (`t_<tid>_g<gen>.dat`) with positional `pread` and sequential scan.
#[derive(Debug)]
pub struct TableDataFile {
    pub path: PathBuf,
    file: File,
    len: u64,
}

impl TableDataFile {
    pub fn open(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;
        let len = file.metadata()?.len();
        Ok(Self { path, file, len })
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn truncate(&mut self, safe_len: u64) -> Result<()> {
        if self.len != safe_len {
            self.file.set_len(safe_len)?;
            self.file.sync_all()?;
            self.len = safe_len;
        }
        Ok(())
    }

    pub fn append_put(&mut self, row_id: RowId, row: &Row) -> Result<(u64, u32)> {
        let mut payload = Vec::new();
        encode_row(row, &mut payload)?;
        let mut frame_buf = Vec::with_capacity(payload.len() + 24);
        let frame_len = encode_data_frame(DataRecordOp::Put, row_id, &payload, &mut frame_buf)?;
        let offset = self.len;
        self.append_bytes(&frame_buf)?;
        Ok((offset, frame_len))
    }

    pub fn append_tombstone(&mut self, row_id: RowId) -> Result<(u64, u32)> {
        let mut frame_buf = Vec::with_capacity(24);
        let frame_len = encode_data_frame(DataRecordOp::Tombstone, row_id, &[], &mut frame_buf)?;
        let offset = self.len;
        self.append_bytes(&frame_buf)?;
        Ok((offset, frame_len))
    }

    pub fn append_bytes(&mut self, bytes: &[u8]) -> Result<u64> {
        let offset = self.len;
        let mut f = &self.file;
        f.seek(SeekFrom::End(0))?;
        f.write_all(bytes)?;
        self.len += bytes.len() as u64;
        Ok(offset)
    }

    /// Flush buffered writes via `BufWriter` in bulk, updating `self.len`.
    pub fn open_batch_writer(&mut self) -> Result<BufWriter<&File>> {
        let mut f = &self.file;
        f.seek(SeekFrom::End(0))?;
        Ok(BufWriter::with_capacity(256 * 1024, &self.file))
    }

    pub fn advance_len(&mut self, added: u64) {
        self.len += added;
    }

    pub fn sync(&self) -> Result<()> {
        self.file.sync_data()?;
        Ok(())
    }

    pub fn sync_all(&self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }

    /// Positional `pread` for a known frame offset and length without changing file cursor.
    pub fn read_frame_at(&self, offset: u64, frame_len: u32) -> Result<DataFrame> {
        let mut buf = vec![0u8; frame_len as usize];
        self.file.read_exact_at(&mut buf, offset)?;
        decode_frame_slice(&buf, offset)
    }

    /// Sequential streaming scan up to `limit_offset` (typically `self.len()`).
    pub fn scan_frames<F>(&self, limit_offset: u64, mut on_frame: F) -> Result<()>
    where
        F: FnMut(DataFrame) -> Result<bool>,
    {
        struct PreadReader<'a> {
            file: &'a File,
            pos: u64,
            limit: u64,
        }
        impl<'a> Read for PreadReader<'a> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.pos >= self.limit {
                    return Ok(0);
                }
                let max_read = ((self.limit - self.pos) as usize).min(buf.len());
                let n = self.file.read_at(&mut buf[..max_read], self.pos)?;
                self.pos += n as u64;
                Ok(n)
            }
        }
        let mut reader = BufReader::with_capacity(
            256 * 1024,
            PreadReader {
                file: &self.file,
                pos: 0,
                limit: limit_offset,
            },
        );
        let mut offset = 0_u64;
        while offset < limit_offset {
            let Some(frame) = read_next_data_frame(&mut reader, offset)? else {
                break;
            };
            offset += frame.frame_len as u64;
            if !on_frame(frame)? {
                break;
            }
        }
        if offset != limit_offset {
            return Err(DbError::new(
                ErrorCode::IoError,
                format!(
                    "data file scan stopped early at {} of {} bytes",
                    offset, limit_offset
                ),
            ));
        }
        Ok(())
    }
}
