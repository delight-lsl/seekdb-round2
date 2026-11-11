use minivectordb::common::error::ErrorCode;
use minivectordb::common::types::{ExecResult, SearchMode};
use minivectordb::recovery::wal::{encode_wal_frame, encode_wal_op, wal_segment_path, KIND_BEGIN, KIND_PUT_ROW};
use minivectordb::sql::executor::Database;
use minivectordb::storage::codec::encode_row;
use minivectordb::common::types::{Row, Value};
use minivectordb::transaction::operation::WalOp;
use serde_json::json;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static TXN_SEQ: AtomicU64 = AtomicU64::new(1);

struct TempDir(PathBuf);
impl TempDir {
    fn new(prefix: &str) -> Self {
        let id = TXN_SEQ.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "minivectordb_txn_{}_{}_{}",
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
fn test_transaction_commit_rollback_nested_begin_and_read_own_writes() {
    let dir = TempDir::new("txn_basic");
    let mut db = Database::open(&dir.0).unwrap();
    db.execute_sql(
        "CREATE TABLE docs (id BIGINT PRIMARY KEY, content VARCHAR(16));",
        SearchMode::Exact,
    )
    .unwrap();
    db.execute_sql("INSERT INTO docs VALUES (1, '初始');", SearchMode::Exact)
        .unwrap();

    // Commit without BEGIN must fail
    assert_eq!(
        db.execute_sql("COMMIT;", SearchMode::Exact)
            .unwrap_err()
            .code,
        ErrorCode::NoActiveTransaction
    );
    assert_eq!(
        db.execute_sql("ROLLBACK;", SearchMode::Exact)
            .unwrap_err()
            .code,
        ErrorCode::NoActiveTransaction
    );

    // BEGIN + nested BEGIN must fail while keeping active txn alive
    db.execute_sql("BEGIN;", SearchMode::Exact).unwrap();
    assert_eq!(
        db.execute_sql("BEGIN;", SearchMode::Exact)
            .unwrap_err()
            .code,
        ErrorCode::TransactionActive
    );

    db.execute_sql(
        "UPDATE docs SET content = '修改' WHERE id = 1;",
        SearchMode::Exact,
    )
    .unwrap();
    // Read own uncommitted modification
    let res = db
        .execute_sql("SELECT content FROM docs WHERE id = 1;", SearchMode::Exact)
        .unwrap();
    assert_eq!(
        res,
        ExecResult::Query {
            columns: vec!["content".into()],
            rows: vec![vec![json!("修改")]],
        }
    );
    db.execute_sql("ROLLBACK;", SearchMode::Exact).unwrap();

    // After rollback, original value is restored
    let res2 = db
        .execute_sql("SELECT content FROM docs WHERE id = 1;", SearchMode::Exact)
        .unwrap();
    assert_eq!(
        res2,
        ExecResult::Query {
            columns: vec!["content".into()],
            rows: vec![vec![json!("初始")]],
        }
    );

    // Now test failed statement inside BEGIN keeps prior changes and commits cleanly
    db.execute_sql("BEGIN;", SearchMode::Exact).unwrap();
    db.execute_sql(
        "UPDATE docs SET content = '保留' WHERE id = 1;",
        SearchMode::Exact,
    )
    .unwrap();
    assert_eq!(
        db.execute_sql(
            "INSERT INTO docs VALUES (2, 'ok'), (1, 'dup');",
            SearchMode::Exact
        )
        .unwrap_err()
        .code,
        ErrorCode::DuplicateKey
    );
    db.execute_sql("COMMIT;", SearchMode::Exact).unwrap();
    db.shutdown().unwrap();

    let mut db2 = Database::open(&dir.0).unwrap();
    let res3 = db2
        .execute_sql("SELECT id, content FROM docs ORDER BY id;", SearchMode::Exact)
        .unwrap();
    assert_eq!(
        res3,
        ExecResult::Query {
            columns: vec!["id".into(), "content".into()],
            rows: vec![vec![json!(1), json!("保留")]],
        }
    );
}

#[test]
fn test_shutdown_rolls_back_uncommitted_transaction() {
    let dir = TempDir::new("txn_shutdown");
    {
        let mut db = Database::open(&dir.0).unwrap();
        db.execute_sql(
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, content VARCHAR(16));",
            SearchMode::Exact,
        )
        .unwrap();
        db.execute_sql(
            "INSERT INTO docs VALUES (1, 'a'), (2, 'b');",
            SearchMode::Exact,
        )
        .unwrap();
        db.execute_sql("BEGIN;", SearchMode::Exact).unwrap();
        db.execute_sql("DELETE FROM docs WHERE id >= 1;", SearchMode::Exact)
            .unwrap();
        db.shutdown().unwrap();
    }

    let mut db2 = Database::open(&dir.0).unwrap();
    let res = db2
        .execute_sql("SELECT id FROM docs ORDER BY id;", SearchMode::Exact)
        .unwrap();
    assert_eq!(
        res,
        ExecResult::Query {
            columns: vec!["id".into()],
            rows: vec![vec![json!(1)], vec![json!(2)]],
        }
    );
}

#[test]
fn test_fault_injection_crash_after_wal_commit_before_datafile_write() {
    let dir = TempDir::new("fault_wal_ahead");
    {
        let mut db = Database::open(&dir.0).unwrap();
        db.execute_sql(
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, content VARCHAR(16));",
            SearchMode::Exact,
        )
        .unwrap();
        db.execute_sql("INSERT INTO docs VALUES (1, 'ckpt');", SearchMode::Exact)
            .unwrap();
        db.shutdown().unwrap(); // creates checkpoint with id=1
    }

    // Reopen, commit id=2 via WAL + data file, THEN simulate crash BEFORE data file flushed
    // by writing a committed txn id=3 directly into WAL without touching data file, PLUS
    // an uncommitted txn id=4 (no COMMIT), PLUS a corrupt partial WAL tail, PLUS corrupt .dat tail!
    {
        let mut db = Database::open(&dir.0).unwrap();
        db.execute_sql("INSERT INTO docs VALUES (2, 'normal');", SearchMode::Exact)
            .unwrap();
        let wal_seq = db.wal.current_seq;
        let dat_path = db.storage.get_table(1).unwrap().file.path.clone();
        drop(db); // drop WITHOUT shutdown() (simulates SIGKILL!)

        // Inject WAL committed txn (id=3) that never reached .dat
        let wal_path = wal_segment_path(&dir.0.join("wal"), wal_seq);
        let mut wal_file = OpenOptions::new().append(true).open(&wal_path).unwrap();
        let mut row3_bytes = Vec::new();
        encode_row(
            &Row {
                values: vec![Value::BigInt(3), Value::Varchar("redo".into())],
            },
            &mut row3_bytes,
        )
        .unwrap();
        let mut buf = Vec::new();
        encode_wal_frame(99, KIND_BEGIN, &[], &mut buf).unwrap();
        let mut op_buf = Vec::new();
        encode_wal_op(
            &WalOp::PutRow {
                table_id: 1,
                row_id: 3,
                old_pk: None,
                row_bytes: row3_bytes,
            },
            &mut op_buf,
        )
        .unwrap();
        encode_wal_frame(99, KIND_PUT_ROW, &op_buf, &mut buf).unwrap();
        encode_wal_frame(99, 6 /* COMMIT */, &[], &mut buf).unwrap();

        // Also inject uncommitted txn (id=4) WITHOUT Commit frame
        encode_wal_frame(100, KIND_BEGIN, &[], &mut buf).unwrap();
        let mut row4_bytes = Vec::new();
        encode_row(
            &Row {
                values: vec![Value::BigInt(4), Value::Varchar("uncomm".into())],
            },
            &mut row4_bytes,
        )
        .unwrap();
        op_buf.clear();
        encode_wal_op(
            &WalOp::PutRow {
                table_id: 1,
                row_id: 4,
                old_pk: None,
                row_bytes: row4_bytes,
            },
            &mut op_buf,
        )
        .unwrap();
        encode_wal_frame(100, KIND_PUT_ROW, &op_buf, &mut buf).unwrap();

        // Also inject truncated corrupt WAL tail bytes
        buf.extend_from_slice(b"WAL1\x01\x02\x03");
        wal_file.write_all(&buf).unwrap();
        wal_file.sync_all().unwrap();

        // Also inject torn garbage into the tail of table .dat file
        let mut dat_file = OpenOptions::new().append(true).open(&dat_path).unwrap();
        dat_file.write_all(b"MVD1CORRUPT_TAIL_BYTES").unwrap();
        dat_file.sync_all().unwrap();
    }

    // Reopen and verify Redo truncates torn .dat and WAL tails, replays id=2 and id=3, drops id=4!
    let mut db_rec = Database::open(&dir.0).unwrap();
    let res = db_rec
        .execute_sql("SELECT id, content FROM docs ORDER BY id;", SearchMode::Exact)
        .unwrap();
    assert_eq!(
        res,
        ExecResult::Query {
            columns: vec!["id".into(), "content".into()],
            rows: vec![
                vec![json!(1), json!("ckpt")],
                vec![json!(2), json!("normal")],
                vec![json!(3), json!("redo")],
            ],
        }
    );

    // Reopen a second time without shutdown to confirm idempotent checkpoint & recovery
    drop(db_rec);
    let mut db_rec2 = Database::open(&dir.0).unwrap();
    let res2 = db_rec2
        .execute_sql("SELECT id, content FROM docs ORDER BY id;", SearchMode::Exact)
        .unwrap();
    assert_eq!(res, res2);
}
