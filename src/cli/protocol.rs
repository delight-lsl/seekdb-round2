use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::{ExecResult, SearchMode};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::borrow::Cow;
use std::io;

pub trait DatabaseBackend {
    fn execute(&mut self, sql: &str, mode: SearchMode) -> Result<ExecResult>;
    fn shutdown(&mut self) -> Result<()>;
}

#[derive(Debug, Deserialize)]
struct Request<'a> {
    id: JsonValue,
    #[serde(borrow)]
    op: Cow<'a, str>,
    #[serde(borrow, default)]
    sql: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    search_mode: Option<Cow<'a, str>>,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    code: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct ErrorResp {
    id: JsonValue,
    ok: bool,
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct OkEmptyResp {
    id: JsonValue,
    ok: bool,
}

#[derive(Debug, Serialize)]
struct OkAffectedResp {
    id: JsonValue,
    ok: bool,
    affected_rows: usize,
}

#[derive(Debug, Serialize)]
struct OkQueryResp {
    id: JsonValue,
    ok: bool,
    columns: Vec<String>,
    rows: Vec<Vec<JsonValue>>,
}

#[repr(C)]
struct PollFd {
    fd: i32,
    events: i16,
    revents: i16,
}

extern "C" {
    fn poll(fds: *mut PollFd, nfds: u64, timeout: i32) -> i32;
    fn sched_yield() -> i32;
    fn read(fd: i32, buf: *mut u8, count: usize) -> isize;
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
}
const F_SETPIPE_SZ: i32 = 1031;
const READ_CHUNK: usize = 256 * 1024;
const DEFAULT_SPIN_US: u64 = 2000;

/// Position of the first `\n` in `h`.
#[inline]
fn find_newline(h: &[u8]) -> Option<usize> {
    let mut i = 0;
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is part of the x86_64 baseline; loads stay inside `h`.
        unsafe {
            let nl = _mm_set1_epi8(b'\n' as i8);
            while i + 16 <= h.len() {
                let v = _mm_loadu_si128(h.as_ptr().add(i) as *const __m128i);
                let m = _mm_movemask_epi8(_mm_cmpeq_epi8(v, nl));
                if m != 0 {
                    return Some(i + m.trailing_zeros() as usize);
                }
                i += 16;
            }
        }
    }
    h[i..].iter().position(|&c| c == b'\n').map(|p| p + i)
}

/// Unbuffered line reader over fd 0 with a large reusable buffer: a request line of
/// any size is pulled from the pipe with as few `read` calls as the pipe allows.
struct StdinLines {
    buf: Vec<u8>,
    pos: usize,
    scanned: usize,
    eof: bool,
    /// How long to busy-poll stdin for the next request before blocking in `read`.
    spin: std::time::Duration,
}

/// Wait for stdin to become readable by polling (yielding the CPU to any runnable task
/// between polls) for at most `limit`. A client that sends the next request right after
/// receiving a response is then served without a sleep/wake-up cycle of this process;
/// when no request arrives within the window the caller blocks in `read` as usual.
fn spin_wait_readable(limit: std::time::Duration) {
    let start = std::time::Instant::now();
    loop {
        let mut p = PollFd {
            fd: 0,
            events: 1,
            revents: 0,
        };
        // SAFETY: `p` is a valid pollfd for the duration of the call.
        if unsafe { poll(&mut p, 1, 0) } != 0 {
            return;
        }
        if start.elapsed() >= limit {
            return;
        }
        // SAFETY: plain syscall.
        unsafe {
            sched_yield();
        }
    }
}

impl StdinLines {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(READ_CHUNK),
            pos: 0,
            scanned: 0,
            eof: false,
            spin: std::time::Duration::from_micros(
                std::env::var("MVDB_SPIN_US")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(DEFAULT_SPIN_US),
            ),
        }
    }

    /// Returns the byte range of the next line (without the terminator), or `None` at EOF.
    fn next_line(&mut self) -> io::Result<Option<(usize, usize)>> {
        loop {
            if let Some(p) = find_newline(&self.buf[self.scanned..]) {
                let start = self.pos;
                let end = self.scanned + p;
                self.pos = end + 1;
                self.scanned = self.pos;
                return Ok(Some((start, end)));
            }
            self.scanned = self.buf.len();
            if self.eof {
                if self.pos < self.buf.len() {
                    // Final line without a terminator.
                    let start = self.pos;
                    let end = self.buf.len();
                    self.pos = end;
                    self.scanned = end;
                    return Ok(Some((start, end)));
                }
                return Ok(None);
            }
            if self.pos > 0 {
                self.buf.drain(..self.pos);
                self.scanned -= self.pos;
                self.pos = 0;
            }
            if self.buf.capacity() > 8 * READ_CHUNK && self.buf.is_empty() {
                self.buf = Vec::with_capacity(READ_CHUNK);
            }
            self.buf.reserve(READ_CHUNK);
            if !self.spin.is_zero() && self.pos == self.buf.len() {
                spin_wait_readable(self.spin);
            }
            let n = loop {
                let spare = self.buf.spare_capacity_mut();
                // SAFETY: `spare` is writable memory of `spare.len()` bytes owned by `buf`.
                let n = unsafe { read(0, spare.as_mut_ptr() as *mut u8, spare.len()) };
                if n < 0 {
                    let e = io::Error::last_os_error();
                    if e.kind() == io::ErrorKind::Interrupted {
                        continue;
                    }
                    return Err(e);
                }
                break n as usize;
            };
            if n == 0 {
                self.eof = true;
            } else {
                // SAFETY: the kernel initialized `n` bytes of the spare capacity.
                unsafe { self.buf.set_len(self.buf.len() + n) };
            }
        }
    }
}

fn write_stdout(mut bytes: &[u8]) -> io::Result<()> {
    while !bytes.is_empty() {
        // SAFETY: `bytes` is a valid readable slice.
        let n = unsafe { write(1, bytes.as_ptr(), bytes.len()) };
        if n < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e);
        }
        bytes = &bytes[n as usize..];
    }
    Ok(())
}

/// Serialize one response line into `out` and send it with a single `write`.
fn emit_json<T: Serialize>(out: &mut Vec<u8>, val: &T) -> io::Result<()> {
    out.clear();
    serde_json::to_writer(&mut *out, val).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    out.push(b'\n');
    let res = write_stdout(out);
    if out.capacity() > 4 * 1024 * 1024 {
        *out = Vec::new();
    }
    res
}

fn error_resp(id: JsonValue, e: DbError) -> ErrorResp {
    ErrorResp {
        id,
        ok: false,
        error: ErrorDetail {
            code: e.code.as_str().to_string(),
            message: e.message,
        },
    }
}

/// Run the JSON Lines stdin/stdout server loop.
pub fn run_stdio_loop(db: &mut impl DatabaseBackend) -> io::Result<()> {
    // A larger pipe lets the driver hand over a big request without stalling on a full pipe.
    // SAFETY: plain fcntl on our own descriptors; failure is harmless.
    unsafe {
        fcntl(0, F_SETPIPE_SZ, 1024 * 1024);
    }
    let mut out: Vec<u8> = Vec::with_capacity(16 * 1024);
    write_stdout(b"{\"type\":\"ready\"}\n")?;

    let mut lines = StdinLines::new();
    loop {
        let Some((start, end)) = lines.next_line()? else {
            // EOF: ensure clean shutdown/rollback of uncommitted transactions
            if let Err(e) = db.shutdown() {
                eprintln!("shutdown on EOF error: {}", e);
            }
            break;
        };
        let mut line = &lines.buf[start..end];
        while let [first, rest @ ..] = line {
            if first.is_ascii_whitespace() {
                line = rest;
            } else {
                break;
            }
        }
        while let [rest @ .., last] = line {
            if last.is_ascii_whitespace() {
                line = rest;
            } else {
                break;
            }
        }
        if line.is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_slice(line) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("malformed JSON request: {}", e);
                let resp = error_resp(
                    JsonValue::Null,
                    DbError::new(ErrorCode::SyntaxError, format!("invalid JSON request: {}", e)),
                );
                emit_json(&mut out, &resp)?;
                continue;
            }
        };

        match &*req.op {
            "shutdown" => {
                if let Err(e) = db.shutdown() {
                    eprintln!("shutdown error: {}", e);
                    emit_json(&mut out, &error_resp(req.id, e))?;
                } else {
                    emit_json(&mut out, &OkEmptyResp { id: req.id, ok: true })?;
                }
                break;
            }
            "execute" => {
                let Some(sql) = req.sql.as_deref() else {
                    let e = DbError::new(
                        ErrorCode::InvalidArgument,
                        "missing 'sql' field in execute request",
                    );
                    emit_json(&mut out, &error_resp(req.id, e))?;
                    continue;
                };
                let mode = match req.search_mode.as_deref() {
                    Some(m) => match SearchMode::parse(m) {
                        Ok(sm) => sm,
                        Err(e) => {
                            emit_json(&mut out, &error_resp(req.id, e))?;
                            continue;
                        }
                    },
                    None => SearchMode::Exact,
                };
                match db.execute(sql, mode) {
                    Ok(ExecResult::Ok) => {
                        emit_json(&mut out, &OkEmptyResp { id: req.id, ok: true })?;
                    }
                    Ok(ExecResult::AffectedRows(n)) => {
                        emit_json(
                            &mut out,
                            &OkAffectedResp {
                                id: req.id,
                                ok: true,
                                affected_rows: n,
                            },
                        )?;
                    }
                    Ok(ExecResult::Query { columns, rows }) => {
                        emit_json(
                            &mut out,
                            &OkQueryResp {
                                id: req.id,
                                ok: true,
                                columns,
                                rows,
                            },
                        )?;
                    }
                    Err(e) => {
                        emit_json(&mut out, &error_resp(req.id, e))?;
                    }
                }
            }
            other => {
                let err = DbError::new(
                    ErrorCode::UnsupportedSql,
                    format!("unsupported protocol op '{}'", other),
                );
                emit_json(&mut out, &error_resp(req.id, err))?;
            }
        }
    }
    Ok(())
}
