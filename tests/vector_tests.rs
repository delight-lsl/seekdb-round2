use minivectordb::common::types::{ExecResult, SearchMode};
use minivectordb::sql::executor::Database;
use minivectordb::vector::distance::{l2_distance, l2_distance_squared};
use minivectordb::vector::index::HnswIndex;
use serde_json::json;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static VEC_SEQ: AtomicU64 = AtomicU64::new(1);

struct TempDir(PathBuf);
impl TempDir {
    fn new(prefix: &str) -> Self {
        let id = VEC_SEQ.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "minivectordb_vec_{}_{}_{}",
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
fn test_simd_and_scalar_distance_equivalence() {
    for dim in [1, 2, 3, 7, 8, 15, 16, 64, 128, 771] {
        let a: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.137).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.271).cos()).collect();
        let expected: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = (x as f64) - (y as f64);
                d * d
            })
            .sum();
        let actual = l2_distance_squared(&a, &b).unwrap();
        assert!(
            (actual - expected).abs() < 1e-10,
            "dim {} mismatch: {} vs {}",
            dim,
            actual,
            expected
        );
        assert!((l2_distance(&a, &b).unwrap() - expected.sqrt()).abs() < 1e-10);
    }
}

#[test]
fn test_vector_index_maintenance_restart_and_ann() {
    let dir = TempDir::new("vec_idx");
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
            "CREATE VECTOR INDEX docs_vec ON docs(embedding);",
            SearchMode::Exact,
        )
        .unwrap();
        db.execute_sql(
            "UPDATE docs SET embedding = '[3,4]' WHERE id = 1;",
            SearchMode::Exact,
        )
        .unwrap();
        db.execute_sql("DELETE FROM docs WHERE id = 2;", SearchMode::Exact)
            .unwrap();
        db.execute_sql(
            "INSERT INTO docs VALUES (4,'甲','四','[0,0]');",
            SearchMode::Exact,
        )
        .unwrap();

        let exact_res = db
            .execute_sql(
                "SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance FROM docs ORDER BY distance LIMIT 10;",
                SearchMode::Exact,
            )
            .unwrap();
        assert_eq!(
            exact_res,
            ExecResult::Query {
                columns: vec!["id".into(), "distance".into()],
                rows: vec![
                    vec![json!(4), json!(0.0)],
                    vec![json!(3), json!(2.0)],
                    vec![json!(1), json!(5.0)],
                ],
            }
        );
        db.shutdown().unwrap();
    }

    {
        let mut db = Database::open(&dir.0).unwrap();
        let res_exact = db
            .execute_sql(
                "SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance FROM docs ORDER BY distance LIMIT 10;",
                SearchMode::Exact,
            )
            .unwrap();
        let res_ann = db
            .execute_sql(
                "SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance FROM docs ORDER BY distance LIMIT 10;",
                SearchMode::Ann,
            )
            .unwrap();
        assert_eq!(res_exact, res_ann);
        assert_eq!(
            res_ann,
            ExecResult::Query {
                columns: vec!["id".into(), "distance".into()],
                rows: vec![
                    vec![json!(4), json!(0.0)],
                    vec![json!(3), json!(2.0)],
                    vec![json!(1), json!(5.0)],
                ],
            }
        );
    }
}

#[test]
fn test_hnsw_recall_and_filtered_ann() {
    let dir = TempDir::new("hnsw_recall");
    let mut db = Database::open(&dir.0).unwrap();
    db.execute_sql(
        "CREATE TABLE items (id BIGINT PRIMARY KEY, grp VARCHAR(8), vec VECTOR(16));",
        SearchMode::Exact,
    )
    .unwrap();

    // Deterministic 1000-vector dataset
    let n = 1000usize;
    let dim = 16usize;
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    let mut next_f32 = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let mut sql = String::from("INSERT INTO items VALUES ");
    for i in 1..=n {
        if i > 1 {
            sql.push(',');
        }
        let grp = if i % 10 == 0 { "rare" } else { "common" };
        let coords: Vec<String> = (0..dim).map(|_| format!("{:.5}", next_f32())).collect();
        sql.push_str(&format!("({}, '{}', '[{}]')", i, grp, coords.join(",")));
    }
    sql.push(';');
    db.execute_sql(&sql, SearchMode::Exact).unwrap();
    db.execute_sql(
        "CREATE VECTOR INDEX items_vec ON items(vec);",
        SearchMode::Exact,
    )
    .unwrap();

    // Unfiltered Top-10 recall check
    let mut hits = 0usize;
    let queries = 10usize;
    let k = 10usize;
    for q in 0..queries {
        let qv: Vec<String> = (0..dim)
            .map(|d| format!("{:.5}", ((q * 7 + d) as f32 * 0.1).sin()))
            .collect();
        let qstr = format!("[{}]", qv.join(","));
        let exact_sql = format!(
            "SELECT id FROM items ORDER BY L2_DISTANCE(vec, '{}') LIMIT {};",
            qstr, k
        );
        let ExecResult::Query { rows: erows, .. } =
            db.execute_sql(&exact_sql, SearchMode::Exact).unwrap()
        else {
            panic!("expected query");
        };
        let ExecResult::Query { rows: arows, .. } =
            db.execute_sql(&exact_sql, SearchMode::Ann).unwrap()
        else {
            panic!("expected query");
        };
        let eset: HashSet<i64> = erows.iter().map(|r| r[0].as_i64().unwrap()).collect();
        let aset: HashSet<i64> = arows.iter().map(|r| r[0].as_i64().unwrap()).collect();
        assert_eq!(aset.len(), k, "ANN returned duplicate or missing IDs");
        hits += eset.intersection(&aset).count();
    }
    let recall = (hits as f64) / ((queries * k) as f64);
    assert!(recall >= 0.95, "HNSW recall {} < 0.95", recall);

    // Filtered Top-10 on 'rare' (10% selectivity)
    let fsql = "SELECT id, L2_DISTANCE(vec, '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]') AS d FROM items WHERE grp = 'rare' ORDER BY d LIMIT 10;";
    let ExecResult::Query { rows: erows, .. } = db.execute_sql(fsql, SearchMode::Exact).unwrap()
    else {
        panic!("expected query");
    };
    let ExecResult::Query { rows: arows, .. } = db.execute_sql(fsql, SearchMode::Ann).unwrap()
    else {
        panic!("expected query");
    };
    assert_eq!(arows.len(), 10);
    let eset: HashSet<i64> = erows.iter().map(|r| r[0].as_i64().unwrap()).collect();
    let aset: HashSet<i64> = arows.iter().map(|r| r[0].as_i64().unwrap()).collect();
    let filtered_recall = (eset.intersection(&aset).count() as f64) / 10.0;
    assert!(
        filtered_recall >= 0.9,
        "Filtered ANN recall {} < 0.9",
        filtered_recall
    );
    // All returned rows must actually belong to 'rare' (id % 10 == 0)
    for id in &aset {
        assert_eq!(id % 10, 0, "Filtered ANN returned non-matching id {}", id);
    }

    // Verify spill-mode threshold transitions to disk .vec + LRU cache + snapshot reload cleanly
    let spill_path = dir.0.join("spill_test.vec");
    let snap_path = dir.0.join("spill_test.hnsw");
    let tmp_path = dir.0.join("spill_test.tmp");
    let mut hnsw = HnswIndex::with_cache_limit("test_spill".into(), 8, spill_path.clone(), 512);
    for i in 1..=100_u64 {
        let v: Vec<f32> = (0..8).map(|d| (i as f32) * 0.1 + (d as f32) * 0.01).collect();
        hnsw.upsert(i, Some(i as i64), &v).unwrap();
    }
    assert!(hnsw.is_spilled(), "HNSW should spill beyond 512-byte cache cap");
    hnsw.save_snapshot(&snap_path, &tmp_path).unwrap();

    let mut reloaded =
        HnswIndex::load_snapshot("test_spill".into(), &snap_path, spill_path).unwrap().unwrap();
    assert!(reloaded.is_spilled());
    let q: Vec<f32> = (0..8).map(|d| 5.0 + (d as f32) * 0.01).collect();
    let (res, _) = reloaded
        .search_knn(&q, 1, &HashSet::new(), None::<&mut fn(u64) -> minivectordb::common::error::Result<bool>>)
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].2, 50);
}
