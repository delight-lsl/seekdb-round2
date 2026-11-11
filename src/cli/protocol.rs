use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::{ExecResult, SearchMode};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::io::{self, BufRead, Write};

pub trait DatabaseBackend {
    fn execute(&mut self, sql: &str, mode: SearchMode) -> Result<ExecResult>;
    fn shutdown(&mut self) -> Result<()>;
}

#[derive(Debug, Deserialize)]
struct Request {
    id: JsonValue,
    op: String,
    #[serde(default)]
    sql: Option<String>,
    #[serde(default)]
    search_mode: Option<String>,
}

#[derive(Debug, Serialize)]
struct ReadyMsg {
    #[serde(rename = "type")]
    msg_type: &'static str,
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

fn emit_json<T: Serialize>(out: &mut impl Write, val: &T) -> io::Result<()> {
    serde_json::to_writer(&mut *out, val).map_err(io::Error::other)?;
    out.write_all(b"\n")?;
    out.flush()
}

/// Run the JSON Lines stdin/stdout server loop.
pub fn run_stdio_loop(db: &mut impl DatabaseBackend) -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    emit_json(&mut out, &ReadyMsg { msg_type: "ready" })?;

    let mut reader = stdin.lock();
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            // EOF: ensure clean shutdown/rollback of uncommitted transactions
            if let Err(e) = db.shutdown() {
                eprintln!("shutdown on EOF error: {}", e);
            }
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("malformed JSON request: {} | line={}", e, trimmed);
                let resp = ErrorResp {
                    id: JsonValue::Null,
                    ok: false,
                    error: ErrorDetail {
                        code: ErrorCode::SyntaxError.as_str().to_string(),
                        message: format!("invalid JSON request: {}", e),
                    },
                };
                emit_json(&mut out, &resp)?;
                continue;
            }
        };

        match req.op.as_str() {
            "shutdown" => {
                if let Err(e) = db.shutdown() {
                    eprintln!("shutdown error: {}", e);
                    let resp = ErrorResp {
                        id: req.id,
                        ok: false,
                        error: ErrorDetail {
                            code: e.code.as_str().to_string(),
                            message: e.message,
                        },
                    };
                    emit_json(&mut out, &resp)?;
                } else {
                    emit_json(&mut out, &OkEmptyResp { id: req.id, ok: true })?;
                }
                break;
            }
            "execute" => {
                let sql = match req.sql.as_deref() {
                    Some(s) => s,
                    None => {
                        let resp = ErrorResp {
                            id: req.id,
                            ok: false,
                            error: ErrorDetail {
                                code: ErrorCode::InvalidArgument.as_str().to_string(),
                                message: "missing 'sql' field in execute request".to_string(),
                            },
                        };
                        emit_json(&mut out, &resp)?;
                        continue;
                    }
                };
                let mode = match req.search_mode.as_deref() {
                    Some(m) => match SearchMode::parse(m) {
                        Ok(sm) => sm,
                        Err(e) => {
                            let resp = ErrorResp {
                                id: req.id,
                                ok: false,
                                error: ErrorDetail {
                                    code: e.code.as_str().to_string(),
                                    message: e.message,
                                },
                            };
                            emit_json(&mut out, &resp)?;
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
                        emit_json(
                            &mut out,
                            &ErrorResp {
                                id: req.id,
                                ok: false,
                                error: ErrorDetail {
                                    code: e.code.as_str().to_string(),
                                    message: e.message,
                                },
                            },
                        )?;
                    }
                }
            }
            other => {
                let err = DbError::new(
                    ErrorCode::UnsupportedSql,
                    format!("unsupported protocol op '{}'", other),
                );
                emit_json(
                    &mut out,
                    &ErrorResp {
                        id: req.id,
                        ok: false,
                        error: ErrorDetail {
                            code: err.code.as_str().to_string(),
                            message: err.message,
                        },
                    },
                )?;
            }
        }
    }
    Ok(())
}
