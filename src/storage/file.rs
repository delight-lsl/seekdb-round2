use crate::common::error::{DbError, ErrorCode, Result};
use crate::storage::record::{
    decode_frame_slice, DataFrame, DataRecordOp, DATA_FRAME_MAGIC, DATA_HEADER_LEN,
    DATA_TRAILER_LEN,
};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    fn sync_file_range(fd: i32, offset: i64, nbytes: i64, flags: u32) -> i32;
}

/// What "durable before the commit is acknowledged" means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// `fdatasync`: data and the metadata needed to read it back are on stable storage
    /// (survives power loss). This is the default.
    Full,
    /// `sync_file_range(write + wait)`: the data blocks are written to the device before
    /// the acknowledgement, without forcing a filesystem journal commit / cache flush.
    /// Survives any process kill; not a power-loss guarantee.
    Range,
    /// Leave writeback to the kernel. Survives a process kill only.
    Off,
}

/// Policy selected by the `MVDB_SYNC` environment variable (`full` | `range` | `off`).
pub fn sync_policy() -> SyncPolicy {
    static POLICY: std::sync::OnceLock<SyncPolicy> = std::sync::OnceLock::new();
    *POLICY.get_or_init(|| match std::env::var("MVDB_SYNC").as_deref() {
        Ok("range") => SyncPolicy::Range,
        Ok("off") => SyncPolicy::Off,
        _ => SyncPolicy::Full,
    })
}

/// Make `[offset, offset + len)` of `file` durable according to the active policy.
pub fn commit_sync(file: &File, offset: u64, len: u64) -> std::io::Result<()> {
    match sync_policy() {
        SyncPolicy::Full => file.sync_data(),
        SyncPolicy::Range => {
            // SAFETY: plain syscall on an open descriptor.
            let r = unsafe { sync_file_range(file.as_raw_fd(), offset as i64, len as i64, 1 | 2 | 4) };
            if r == 0 {
                Ok(())
            } else {
                file.sync_data()
            }
        }
        SyncPolicy::Off => Ok(()),
    }
}
const POSIX_FADV_RANDOM: i32 = 1;
const POSIX_FADV_DONTNEED: i32 = 4;
const PAGE: u64 = 4096;
const SCAN_CHUNK: usize = 256 * 1024;

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

/// Outcome of scanning a table file up to its last committed group.
#[derive(Debug, Clone, Copy)]
pub struct ScanSummary {
    /// End offset of the last complete, committed group.
    pub valid_len: u64,
}

/// Append-only table heap file (`t_<tid>_g<gen>.dat`).
///
/// The file is the only durable copy of the rows: every transaction appends its
/// row frames followed by a terminator frame and is made durable with a single
/// `fdatasync`. A group without a valid terminator is an uncommitted tail and is
/// truncated on recovery.
#[derive(Debug)]
pub struct TableDataFile {
    pub path: PathBuf,
    file: File,
    len: u64,
    /// Page-cache bytes below this offset were already released.
    dropped_upto: u64,
    /// Whether the file was newly created by `open` (directory entry not yet synced).
    pub created: bool,
}

impl TableDataFile {
    pub fn open(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let created = !path.exists();
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            path,
            file,
            len,
            dropped_upto: 0,
            created,
        })
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn truncate(&mut self, safe_len: u64) -> Result<()> {
        if self.len != safe_len {
            self.file.set_len(safe_len)?;
            self.file.sync_all()?;
            self.len = safe_len;
            self.dropped_upto = self.dropped_upto.min(safe_len);
        }
        Ok(())
    }

    /// Append one committed group and make it durable before returning.
    /// `drop_cache` releases the now clean page-cache pages (the rows stay in the
    /// in-memory table, so they are never read back from this file).
    pub fn commit_append(&mut self, bytes: &[u8], drop_cache: bool) -> Result<u64> {
        let offset = self.len;
        let res = self
            .file
            .write_all_at(bytes, offset)
            .and_then(|_| commit_sync(&self.file, offset, bytes.len() as u64));
        if let Err(e) = res {
            // Leave no half-written group behind a failed statement.
            let _ = self.file.set_len(offset);
            return Err(e.into());
        }
        self.len = offset + bytes.len() as u64;
        if drop_cache {
            self.drop_page_cache();
        }
        Ok(offset)
    }

    /// Append without syncing (used by compaction, which syncs once at the end).
    pub fn append_unsynced(&mut self, bytes: &[u8]) -> Result<u64> {
        let offset = self.len;
        self.file.write_all_at(bytes, offset)?;
        self.len += bytes.len() as u64;
        Ok(offset)
    }

    /// Release clean page-cache pages of the fully written prefix. The partially
    /// filled tail page is kept so that the next append does not read it back.
    pub fn drop_page_cache(&mut self) {
        let upto = self.len / PAGE * PAGE;
        if upto > self.dropped_upto {
            // SAFETY: plain syscall on an owned, open descriptor.
            unsafe {
                posix_fadvise(
                    self.file.as_raw_fd(),
                    self.dropped_upto as i64,
                    (upto - self.dropped_upto) as i64,
                    POSIX_FADV_DONTNEED,
                );
            }
            self.dropped_upto = upto;
        }
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

    /// Sequential scan of `[0, limit_offset)` delivering borrowed frames
    /// `(op, row_id, payload, frame_offset, frame_len)`.
    ///
    /// With `tolerant` set, a truncated or corrupt frame ends the scan quietly
    /// (recovery decides what to keep); otherwise it is reported as an error.
    /// With `drop_cache` set, page-cache pages are released as soon as they are consumed
    /// so that scanning a file does not leave a second copy of it in memory.
    /// Returns the offset just past the last frame handed to the callback.
    pub fn scan_raw<F>(
        &self,
        limit_offset: u64,
        tolerant: bool,
        drop_cache: bool,
        mut on_frame: F,
    ) -> Result<u64>
    where
        F: FnMut(DataRecordOp, u64, &[u8], u64, u32) -> Result<bool>,
    {
        let fd = self.file.as_raw_fd();
        if drop_cache {
            // SAFETY: plain syscall on an owned, open descriptor.
            unsafe {
                posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
            }
        }
        let mut buf: Vec<u8> = Vec::with_capacity(SCAN_CHUNK);
        let mut buf_start = 0_u64; // file offset of buf[0]
        let mut pos = 0_usize; // parse position inside buf
        let mut file_pos = 0_u64; // next file offset to read
        let mut dropped = 0_u64;
        let mut stopped_early = false;
        let mut consumed = 0_u64;

        loop {
            let offset = buf_start + pos as u64;
            if offset >= limit_offset {
                break;
            }
            let avail = buf.len() - pos;
            let mut need = DATA_HEADER_LEN + DATA_TRAILER_LEN;
            if avail >= DATA_HEADER_LEN {
                let h = &buf[pos..pos + DATA_HEADER_LEN];
                if h[0..4] != DATA_FRAME_MAGIC {
                    if tolerant {
                        break;
                    }
                    return Err(DbError::new(ErrorCode::IoError, "corrupted data frame magic"));
                }
                let payload_len = u32::from_le_bytes([h[13], h[14], h[15], h[16]]) as usize;
                need = DATA_HEADER_LEN + payload_len + DATA_TRAILER_LEN;
                if offset + need as u64 > limit_offset {
                    if tolerant {
                        break;
                    }
                    return Err(DbError::new(ErrorCode::IoError, "truncated data frame"));
                }
            }
            if avail < need {
                // Refill: keep the unparsed remainder, then read more.
                if pos > 0 {
                    buf.drain(..pos);
                    buf_start += pos as u64;
                    pos = 0;
                }
                if file_pos >= limit_offset {
                    if tolerant {
                        break;
                    }
                    return Err(DbError::new(ErrorCode::IoError, "truncated data frame header"));
                }
                let want = need.max(SCAN_CHUNK).min((limit_offset - file_pos) as usize + buf.len());
                let old_len = buf.len();
                if want > old_len {
                    buf.resize(want, 0);
                    let mut got = old_len;
                    while got < want {
                        let n = self.file.read_at(&mut buf[got..want], file_pos)?;
                        if n == 0 {
                            break;
                        }
                        got += n;
                        file_pos += n as u64;
                    }
                    buf.truncate(got);
                }
                if drop_cache {
                    let upto = file_pos / PAGE * PAGE;
                    if upto > dropped {
                        // SAFETY: plain syscall on an owned, open descriptor.
                        unsafe {
                            posix_fadvise(fd, dropped as i64, (upto - dropped) as i64, POSIX_FADV_DONTNEED);
                        }
                        dropped = upto;
                    }
                }
                if buf.len() - pos < need {
                    if tolerant {
                        break;
                    }
                    return Err(DbError::new(ErrorCode::IoError, "truncated data frame"));
                }
                continue;
            }
            let frame = &buf[pos..pos + need];
            let body_end = need - DATA_TRAILER_LEN;
            let expected_crc = u32::from_le_bytes([
                frame[body_end],
                frame[body_end + 1],
                frame[body_end + 2],
                frame[body_end + 3],
            ]);
            if crc32fast::hash(&frame[..body_end]) != expected_crc {
                if tolerant {
                    break;
                }
                return Err(DbError::new(ErrorCode::IoError, "corrupted data frame CRC32"));
            }
            let op = match DataRecordOp::from_u8(frame[4]) {
                Ok(op) => op,
                Err(e) => {
                    if tolerant {
                        break;
                    }
                    return Err(e);
                }
            };
            let row_id = u64::from_le_bytes([
                frame[5], frame[6], frame[7], frame[8], frame[9], frame[10], frame[11], frame[12],
            ]);
            pos += need;
            consumed = offset + need as u64;
            if !on_frame(op, row_id, &frame[DATA_HEADER_LEN..body_end], offset, need as u32)? {
                stopped_early = true;
                break;
            }
        }
        if !tolerant && !stopped_early && consumed != limit_offset && limit_offset != 0 {
            return Err(DbError::new(
                ErrorCode::IoError,
                format!("data file scan stopped early at {} of {} bytes", consumed, limit_offset),
            ));
        }
        Ok(consumed)
    }

    /// Sequential streaming scan of row frames (`Put` / `Tombstone`) up to `limit_offset`.
    /// Group terminator frames are skipped.
    pub fn scan_frames<F>(&self, limit_offset: u64, mut on_frame: F) -> Result<()>
    where
        F: FnMut(DataFrame) -> Result<bool>,
    {
        self.scan_raw(limit_offset, false, false, |op, row_id, payload, off, len| {
            if op != DataRecordOp::Put && op != DataRecordOp::Tombstone {
                return Ok(true);
            }
            on_frame(DataFrame {
                op,
                row_id,
                payload: payload.to_vec(),
                frame_offset: off,
                frame_len: len,
            })
        })?;
        Ok(())
    }
}
