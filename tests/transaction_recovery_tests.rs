use minivectordb::common::error::ErrorCode;
use minivectordb::common::types::{ExecResult, SearchMode};
use minivectordb::recovery::wal::{encode_wal_frame, wal_segment_path, KIND_BEGIN};
use minivectordb::sql::executor::Database;
use minivectordb::storage::codec::encode_row;
use minivectordb::common::types::{Row, Value};
use minivectordb::storage::record::{begin_data_frame, finish_data_frame, DataRecordOp};
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
fn test_fault_injection_uncommitted_and_torn_tails_are_discarded() {
    let dir = TempDir::new("fault_tails");
    {
        let mut db = Database::open(&dir.0).unwrap();
        db.execute_sql(
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, content VARCHAR(16));",
            SearchMode::Exact,
        )
        .unwrap();
        db.execute_sql("INSERT INTO docs VALUES (1, 'first');", SearchMode::Exact)
            .unwrap();
        db.shutdown().unwrap();
    }

    {
        let mut db = Database::open(&dir.0).unwrap();
        db.execute_sql("INSERT INTO docs VALUES (2, 'normal');", SearchMode::Exact)
            .unwrap();
        let wal_seq = db.wal.current_seq;
        let dat_path = db.storage.get_table(1).unwrap().file.path.clone();
        drop(db); // drop WITHOUT shutdown() (simulates SIGKILL)

        let put_frame = |row_id: u64, id: i64, text: &str, out: &mut Vec<u8>| {
            let start = begin_data_frame(DataRecordOp::Put, row_id, out);
            encode_row(
                &Row {
                    values: vec![Value::BigInt(id), Value::Varchar(text.into())],
                },
                out,
            )
            .unwrap();
            finish_data_frame(start, out).unwrap();
        };

        let mut buf = Vec::new();
        // A group staged by a multi-table transaction (txn 99) whose commit record never
        // reached the WAL: complete frames, complete terminator, still not committed.
        put_frame(3, 3, "pending", &mut buf);
        let start = begin_data_frame(DataRecordOp::Pending, 99, &mut buf);
        finish_data_frame(start, &mut buf).unwrap();
        // A self-committing group cut off before its terminator.
        put_frame(4, 4, "no-term", &mut buf);
        // Torn garbage at the very end.
        buf.extend_from_slice(b"MVD1CORRUPT_TAIL_BYTES");
        let mut dat_file = OpenOptions::new().append(true).open(&dat_path).unwrap();
        dat_file.write_all(&buf).unwrap();
        dat_file.sync_all().unwrap();

        // A transaction in the WAL without a COMMIT frame, followed by a torn frame.
        let wal_path = wal_segment_path(&dir.0.join("wal"), wal_seq);
        let mut wal_file = OpenOptions::new().append(true).open(&wal_path).unwrap();
        let mut wbuf = Vec::new();
        encode_wal_frame(99, KIND_BEGIN, &[], &mut wbuf).unwrap();
        wbuf.extend_from_slice(b"WAL2\x01\x02\x03");
        wal_file.write_all(&wbuf).unwrap();
        wal_file.sync_all().unwrap();
    }

    let expected = ExecResult::Query {
        columns: vec!["id".into(), "content".into()],
        rows: vec![
            vec![json!(1), json!("first")],
            vec![json!(2), json!("normal")],
        ],
    };
    let mut db_rec = Database::open(&dir.0).unwrap();
    let res = db_rec
        .execute_sql("SELECT id, content FROM docs ORDER BY id;", SearchMode::Exact)
        .unwrap();
    assert_eq!(res, expected);

    // New commits after the cut-off tail must survive another crash-style reopen.
    db_rec
        .execute_sql("INSERT INTO docs VALUES (3, 'after');", SearchMode::Exact)
        .unwrap();
    drop(db_rec);
    let mut db_rec2 = Database::open(&dir.0).unwrap();
    let res2 = db_rec2
        .execute_sql("SELECT id, content FROM docs ORDER BY id;", SearchMode::Exact)
        .unwrap();
    assert_eq!(
        res2,
        ExecResult::Query {
            columns: vec!["id".into(), "content".into()],
            rows: vec![
                vec![json!(1), json!("first")],
                vec![json!(2), json!("normal")],
                vec![json!(3), json!("after")],
            ],
        }
    );
}

#[test]
fn test_multi_table_transaction_is_atomic_across_restart() {
    let dir = TempDir::new("multi_table_txn");
    {
        let mut db = Database::open(&dir.0).unwrap();
        for sql in [
            "CREATE TABLE a (id BIGINT PRIMARY KEY, v VECTOR(2));",
            "CREATE TABLE b (id BIGINT PRIMARY KEY, s VARCHAR(8));",
            "BEGIN;",
            "INSERT INTO a VALUES (1, '[1, 2]'), (2, '[3, 4]');",
            "INSERT INTO b VALUES (10, 'x');",
            "CREATE TABLE c (id BIGINT PRIMARY KEY);",
            "INSERT INTO c VALUES (7);",
            "COMMIT;",
            "BEGIN;",
            "INSERT INTO a VALUES (3, '[5, 6]');",
            "INSERT INTO b VALUES (11, 'y');",
        ] {
            db.execute_sql(sql, SearchMode::Exact).unwrap();
        }
        drop(db); // crash with the second transaction still open
    }
    let mut db = Database::open(&dir.0).unwrap();
    for (sql, ids) in [
        ("SELECT id FROM a ORDER BY id;", vec![1, 2]),
        ("SELECT id FROM b ORDER BY id;", vec![10]),
        ("SELECT id FROM c ORDER BY id;", vec![7]),
    ] {
        let res = db.execute_sql(sql, SearchMode::Exact).unwrap();
        assert_eq!(
            res,
            ExecResult::Query {
                columns: vec!["id".into()],
                rows: ids.into_iter().map(|i| vec![json!(i)]).collect(),
            }
        );
    }
}
