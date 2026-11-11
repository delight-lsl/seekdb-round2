use minivectordb::common::error::ErrorCode;
use minivectordb::common::types::{ExecResult, SearchMode};
use minivectordb::sql::executor::Database;
use serde_json::json;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_SEQ: AtomicU64 = AtomicU64::new(1);

struct TempDir(PathBuf);
impl TempDir {
    fn new(prefix: &str) -> Self {
        let id = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "minivectordb_{}_{}_{}",
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
fn test_sql_crud_unicode_bigint_and_pk() {
    let dir = TempDir::new("sql_crud");
    let mut db = Database::open(&dir.0).unwrap();
    assert_eq!(
        db.execute_sql(
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, content VARCHAR(16));",
            SearchMode::Exact
        )
        .unwrap(),
        ExecResult::Ok
    );
    assert_eq!(
        db.execute_sql(
            "INSERT INTO docs VALUES (1, '初始'), (9007199254740993, '中文测试');",
            SearchMode::Exact
        )
        .unwrap(),
        ExecResult::AffectedRows(2)
    );

    let err = db
        .execute_sql("INSERT INTO docs VALUES (1, '重复');", SearchMode::Exact)
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::DuplicateKey);

    let err_len = db
        .execute_sql(
            "INSERT INTO docs VALUES (2, '一二三四五六七八九十十一十二三四五六七');",
            SearchMode::Exact,
        )
        .unwrap_err();
    assert_eq!(err_len.code, ErrorCode::ValueTooLong);

    let res = db
        .execute_sql(
            "SELECT id, content FROM docs ORDER BY id DESC;",
            SearchMode::Exact,
        )
        .unwrap();
    assert_eq!(
        res,
        ExecResult::Query {
            columns: vec!["id".into(), "content".into()],
            rows: vec![
                vec![json!(9007199254740993_i64), json!("中文测试")],
                vec![json!(1), json!("初始")],
            ],
        }
    );
}

#[test]
fn test_sql_unsupported_and_where_ordering() {
    let dir = TempDir::new("sql_where");
    let mut db = Database::open(&dir.0).unwrap();
    db.execute_sql(
        "CREATE TABLE docs (id BIGINT PRIMARY KEY, value BIGINT);",
        SearchMode::Exact,
    )
    .unwrap();
    db.execute_sql(
        "INSERT INTO docs VALUES (3, 30), (1, 10), (2, 20);",
        SearchMode::Exact,
    )
    .unwrap();

    let err = db
        .execute_sql("ALTER TABLE docs ADD extra BIGINT;", SearchMode::Exact)
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::UnsupportedSql);

    let res = db
        .execute_sql(
            "SELECT id FROM docs WHERE value >= 20 AND value < 30;",
            SearchMode::Exact,
        )
        .unwrap();
    assert_eq!(
        res,
        ExecResult::Query {
            columns: vec!["id".into()],
            rows: vec![vec![json!(2)]],
        }
    );

    let res2 = db
        .execute_sql(
            "SELECT id FROM docs ORDER BY value DESC LIMIT 2;",
            SearchMode::Exact,
        )
        .unwrap();
    assert_eq!(
        res2,
        ExecResult::Query {
            columns: vec!["id".into()],
            rows: vec![vec![json!(3)], vec![json!(2)]],
        }
    );
}

#[test]
fn test_vector_validation_and_ties() {
    let dir = TempDir::new("sql_vec");
    let mut db = Database::open(&dir.0).unwrap();
    db.execute_sql(
        "CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), embedding VECTOR(2));",
        SearchMode::Exact,
    )
    .unwrap();

    assert_eq!(
        db.execute_sql(
            "INSERT INTO docs VALUES (1, 'a', '[NaN, 0]');",
            SearchMode::Exact
        )
        .unwrap_err()
        .code,
        ErrorCode::InvalidVector
    );
    assert_eq!(
        db.execute_sql(
            "INSERT INTO docs VALUES (1, 'a', '[1e40, 0]');",
            SearchMode::Exact
        )
        .unwrap_err()
        .code,
        ErrorCode::InvalidVector
    );
    assert_eq!(
        db.execute_sql("INSERT INTO docs VALUES (1, 'a', '[0]');", SearchMode::Exact)
            .unwrap_err()
            .code,
        ErrorCode::DimensionMismatch
    );

    db.execute_sql(
        "INSERT INTO docs VALUES (8,'keep','[1,0]'), (3,'keep','[-1,0]'), (1,'skip','[0,0]');",
        SearchMode::Exact,
    )
    .unwrap();

    let res = db
        .execute_sql(
            "SELECT id, L2_DISTANCE(embedding, '[0,0]') AS distance FROM docs WHERE category = 'keep' ORDER BY distance LIMIT 2;",
            SearchMode::Exact,
        )
        .unwrap();
    assert_eq!(
        res,
        ExecResult::Query {
            columns: vec!["id".into(), "distance".into()],
            rows: vec![vec![json!(3), json!(1.0)], vec![json!(8), json!(1.0)]],
        }
    );
}

#[test]
fn test_batch_insert_atomicity_and_i64_min() {
    let dir = TempDir::new("sql_min");
    let mut db = Database::open(&dir.0).unwrap();
    db.execute_sql(
        "CREATE TABLE docs (id BIGINT PRIMARY KEY, embedding VECTOR(2));",
        SearchMode::Exact,
    )
    .unwrap();
    db.execute_sql(
        "INSERT INTO docs VALUES (-9223372036854775808, '[0,0]');",
        SearchMode::Exact,
    )
    .unwrap();

    let err = db
        .execute_sql(
            "INSERT INTO docs VALUES (2, '[1,1]'), (3, '[1]');",
            SearchMode::Exact,
        )
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::DimensionMismatch);

    let res = db
        .execute_sql("SELECT id FROM docs ORDER BY id;", SearchMode::Exact)
        .unwrap();
    assert_eq!(
        res,
        ExecResult::Query {
            columns: vec!["id".into()],
            rows: vec![vec![json!(-9223372036854775808_i64)]],
        }
    );
}

#[test]
fn test_table_level_pk_not_null_in_and_between() {
    let dir = TempDir::new("sql_ext");
    let mut db = Database::open(&dir.0).unwrap();
    db.execute_sql(
        "CREATE TABLE items (id BIGINT NOT NULL, score BIGINT, label VARCHAR(16), PRIMARY KEY (id));",
        SearchMode::Exact,
    )
    .unwrap();
    db.execute_sql(
        "INSERT INTO items VALUES (1, 10, 'a'), (2, 20, 'b'), (3, 30, 'c'), (4, 40, 'd');",
        SearchMode::Exact,
    )
    .unwrap();

    let res = db
        .execute_sql(
            "SELECT id FROM items WHERE score BETWEEN 20 AND 30 AND id IN (2, 3, 4) ORDER BY id;",
            SearchMode::Exact,
        )
        .unwrap();
    assert_eq!(
        res,
        ExecResult::Query {
            columns: vec!["id".into()],
            rows: vec![vec![json!(2)], vec![json!(3)]],
        }
    );

    assert_eq!(
        db.execute_sql("SELECT COUNT(*) FROM items;", SearchMode::Exact)
            .unwrap_err()
            .code,
        ErrorCode::UnsupportedSql
    );
    assert_eq!(
        db.execute_sql(
            "SELECT id FROM items WHERE id = (SELECT id FROM items LIMIT 1);",
            SearchMode::Exact
        )
        .unwrap_err()
        .code,
        ErrorCode::UnsupportedSql
    );
}
