use minivectordb::common::error::ErrorCode;
use minivectordb::common::types::{ExecResult, SearchMode};
use minivectordb::sql::executor::Database;
use serde_json::json;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static ST_SEQ: AtomicU64 = AtomicU64::new(1);

struct TempDir(PathBuf);
impl TempDir {
    fn new(prefix: &str) -> Self {
        let id = ST_SEQ.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "minivectordb_st_{}_{}_{}",
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
fn test_persistence_restart_mutations_and_multiple_tables() {
    let dir = TempDir::new("restart");
    {
        let mut db = Database::open(&dir.0).unwrap();
        db.execute_sql(
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), content VARCHAR(16), embedding VECTOR(2));",
            SearchMode::Exact,
        )
        .unwrap();
        db.execute_sql(
            "INSERT INTO docs VALUES (3, '乙', '三', '[0, 2]'), (1, '甲', '一', '[0, 0]'), (2, '甲', '二', '[1, 0]');",
            SearchMode::Exact,
        )
        .unwrap();
        db.execute_sql(
            "UPDATE docs SET embedding = '[3,4]', content = '修改' WHERE id = 1;",
            SearchMode::Exact,
        )
        .unwrap();
        db.execute_sql("DELETE FROM docs WHERE id = 2;", SearchMode::Exact)
            .unwrap();
        db.execute_sql(
            "CREATE TABLE tags (id BIGINT PRIMARY KEY, tag VARCHAR(8));",
            SearchMode::Exact,
        )
        .unwrap();
        db.execute_sql("INSERT INTO tags VALUES (1, '独立');", SearchMode::Exact)
            .unwrap();
        db.shutdown().unwrap();
    }

    // Reopen and verify all mutations and multiple tables persisted
    {
        let mut db = Database::open(&dir.0).unwrap();
        let res1 = db
            .execute_sql(
                "SELECT id, content FROM docs ORDER BY id;",
                SearchMode::Exact,
            )
            .unwrap();
        assert_eq!(
            res1,
            ExecResult::Query {
                columns: vec!["id".into(), "content".into()],
                rows: vec![
                    vec![json!(1), json!("修改")],
                    vec![json!(3), json!("三")],
                ],
            }
        );

        let res2 = db
            .execute_sql(
                "SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance FROM docs ORDER BY distance LIMIT 10;",
                SearchMode::Exact,
            )
            .unwrap();
        assert_eq!(
            res2,
            ExecResult::Query {
                columns: vec!["id".into(), "distance".into()],
                rows: vec![
                    vec![json!(3), json!(2.0)],
                    vec![json!(1), json!(5.0)],
                ],
            }
        );

        let res3 = db
            .execute_sql("SELECT id, tag FROM tags;", SearchMode::Exact)
            .unwrap();
        assert_eq!(
            res3,
            ExecResult::Query {
                columns: vec!["id".into(), "tag".into()],
                rows: vec![vec![json!(1), json!("独立")]],
            }
        );

        // Deleted PK (2) can be re-inserted cleanly; existing PK (1) still rejects duplicate
        let err = db
            .execute_sql(
                "INSERT INTO docs VALUES (1, 'x', 'y', '[0,0]');",
                SearchMode::Exact,
            )
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::DuplicateKey);
        db.execute_sql(
            "INSERT INTO docs VALUES (2, '甲', '重生', '[0,1]');",
            SearchMode::Exact,
        )
        .unwrap();
    }
}

#[test]
fn test_compaction_across_generations() {
    let dir = TempDir::new("compact");
    let mut db = Database::open(&dir.0).unwrap();
    db.execute_sql(
        "CREATE TABLE items (id BIGINT PRIMARY KEY, val VARCHAR(32));",
        SearchMode::Exact,
    )
    .unwrap();
    for i in 1..=200 {
        db.execute_sql(
            &format!("INSERT INTO items VALUES ({}, 'v{}');", i, i),
            SearchMode::Exact,
        )
        .unwrap();
    }
    for i in 1..=150 {
        db.execute_sql(
            &format!("DELETE FROM items WHERE id = {};", i),
            SearchMode::Exact,
        )
        .unwrap();
    }
    let dt = db.storage.get_table_mut(1).unwrap();
    let old_path = dt.compact_to_next_generation(&dir.0).unwrap();
    db.storage.save_checkpoint_snapshot(0, 0, 1).unwrap();
    std::fs::remove_file(old_path).unwrap();
    db.shutdown().unwrap();

    let mut db2 = Database::open(&dir.0).unwrap();
    let res = db2
        .execute_sql(
            "SELECT id, val FROM items ORDER BY id LIMIT 3;",
            SearchMode::Exact,
        )
        .unwrap();
    assert_eq!(
        res,
        ExecResult::Query {
            columns: vec!["id".into(), "val".into()],
            rows: vec![
                vec![json!(151), json!("v151")],
                vec![json!(152), json!("v152")],
                vec![json!(153), json!("v153")],
            ],
        }
    );
}
