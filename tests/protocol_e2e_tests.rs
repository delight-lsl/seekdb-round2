use minivectordb::common::types::SearchMode;
use minivectordb::sql::executor::Database;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

static E2E_SEQ: AtomicU64 = AtomicU64::new(1);

struct TempDir(PathBuf);
impl TempDir {
    fn new(prefix: &str) -> Self {
        let id = E2E_SEQ.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "minivectordb_e2e_{}_{}_{}",
            prefix,
            std::process::id(),
            id
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn test_stdio_protocol_handshake_execute_restart_and_errors() {
    let dir = TempDir::new("stdio_e2e");
    let bin = env!("CARGO_BIN_EXE_minivectordb");

    // Session 1: verify ready arrives BEFORE any stdin input, run SQL + error + BEGIN (uncommitted) + shutdown
    {
        let mut child = Command::new(bin)
            .arg("--data-dir")
            .arg(&dir.0)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();

        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let mut stdin = child.stdin.take().unwrap();

        let mut ready_line = String::new();
        stdout.read_line(&mut ready_line).unwrap();
        let ready_json: serde_json::Value = serde_json::from_str(ready_line.trim()).unwrap();
        assert_eq!(ready_json, serde_json::json!({"type": "ready"}));

        let send_req = |stdin: &mut std::process::ChildStdin,
                        stdout: &mut BufReader<std::process::ChildStdout>,
                        req: &serde_json::Value|
         -> serde_json::Value {
            writeln!(stdin, "{}", req).unwrap();
            stdin.flush().unwrap();
            let mut resp = String::new();
            stdout.read_line(&mut resp).unwrap();
            serde_json::from_str(resp.trim()).unwrap()
        };

        let r1 = send_req(
            &mut stdin,
            &mut stdout,
            &serde_json::json!({"id": 1, "op": "execute", "sql": "CREATE TABLE docs (id BIGINT PRIMARY KEY, content VARCHAR(16), vec VECTOR(2));", "search_mode": "exact"}),
        );
        assert_eq!(r1, serde_json::json!({"id": 1, "ok": true}));

        let r2 = send_req(
            &mut stdin,
            &mut stdout,
            &serde_json::json!({"id": "req-2", "op": "execute", "sql": "INSERT INTO docs VALUES (1, '你好', '[3,4]');", "search_mode": "exact"}),
        );
        assert_eq!(
            r2,
            serde_json::json!({"id": "req-2", "ok": true, "affected_rows": 1})
        );

        let r3 = send_req(
            &mut stdin,
            &mut stdout,
            &serde_json::json!({"id": 3, "op": "execute", "sql": "ALTER TABLE docs ADD x BIGINT;", "search_mode": "exact"}),
        );
        assert_eq!(r3["id"], 3);
        assert_eq!(r3["ok"], false);
        assert_eq!(r3["error"]["code"], "unsupported_sql");

        // Uncommitted BEGIN must roll back on shutdown
        let r4 = send_req(
            &mut stdin,
            &mut stdout,
            &serde_json::json!({"id": 4, "op": "execute", "sql": "BEGIN;", "search_mode": "exact"}),
        );
        assert_eq!(r4, serde_json::json!({"id": 4, "ok": true}));
        let r5 = send_req(
            &mut stdin,
            &mut stdout,
            &serde_json::json!({"id": 5, "op": "execute", "sql": "DELETE FROM docs WHERE id = 1;", "search_mode": "exact"}),
        );
        assert_eq!(
            r5,
            serde_json::json!({"id": 5, "ok": true, "affected_rows": 1})
        );

        let r6 = send_req(
            &mut stdin,
            &mut stdout,
            &serde_json::json!({"id": 6, "op": "shutdown"}),
        );
        assert_eq!(r6, serde_json::json!({"id": 6, "ok": true}));
        drop(stdin);

        let mut trailing = String::new();
        stdout.read_line(&mut trailing).unwrap();
        assert!(trailing.is_empty(), "extra stdout after shutdown: {:?}", trailing);
        let status = child.wait().unwrap();
        assert!(status.success());
    }

    // Session 2: reopen and confirm id=1 survived and uncommitted DELETE rolled back
    {
        let mut db = Database::open(&dir.0).unwrap();
        let res = db
            .execute_sql(
                "SELECT id, content, L2_DISTANCE(vec, '[0,0]') AS d FROM docs WHERE L2_DISTANCE(vec, '[0,0]') <= 5.0;",
                SearchMode::Exact,
            )
            .unwrap();
        assert_eq!(
            res,
            minivectordb::common::types::ExecResult::Query {
                columns: vec!["id".into(), "content".into(), "d".into()],
                rows: vec![vec![
                    serde_json::json!(1),
                    serde_json::json!("你好"),
                    serde_json::json!(5.0),
                ]],
            }
        );
    }
}
