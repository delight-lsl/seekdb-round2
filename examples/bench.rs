use minivectordb::common::types::{ExecResult, SearchMode};
use minivectordb::sql::executor::Database;
use minivectordb::vector::distance::l2_distance_squared;
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    let bench_dir = std::env::temp_dir().join(format!("minivectordb_bench_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&bench_dir);
    std::fs::create_dir_all(&bench_dir).unwrap();

    // 1. L2 Distance throughput benchmark (128-dim, 1M evaluations)
    let dim = 128usize;
    let a: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.13).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.29).cos()).collect();
    let t0 = Instant::now();
    let mut acc = 0.0_f64;
    for _ in 0..1_000_000 {
        acc += l2_distance_squared(&a, &b).unwrap();
    }
    let l2_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[Bench 1] L2_distance_squared 1M x 128d: {:.2} ms (checksum={:.2})",
        l2_ms, acc
    );

    // 2. Section 3.3 workload: 100 batches x 100 rows = 10,000 x 128-dim auto-commit inserts
    let n = 10_000usize;
    let mut db = Database::open(&bench_dir).unwrap();
    db.execute_sql(
        "CREATE TABLE vecs (id BIGINT PRIMARY KEY, cat BIGINT, emb VECTOR(128));",
        SearchMode::Exact,
    )
    .unwrap();

    let mut state: u64 = 0xCAFE_BABE_1234_5678;
    let mut next_f = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let t_ins = Instant::now();
    for chunk_start in (1..=n).step_by(100) {
        let mut sql = String::from("INSERT INTO vecs VALUES ");
        let chunk_end = (chunk_start + 99).min(n);
        for id in chunk_start..=chunk_end {
            if id > chunk_start {
                sql.push(',');
            }
            let coords: Vec<String> = (0..dim).map(|_| format!("{:.4}", next_f())).collect();
            sql.push_str(&format!("({}, {}, '[{}]')", id, id % 10, coords.join(",")));
        }
        sql.push(';');
        db.execute_sql(&sql, SearchMode::Exact).unwrap();
    }
    let ins_ms = t_ins.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[Bench 2] 100 batches x 100 rows (10k x 128d) + WAL fsync: {:.2} ms ({:.0} rows/sec)",
        ins_ms,
        (n as f64) / (ins_ms / 1000.0)
    );

    let t_idx = Instant::now();
    db.execute_sql(
        "CREATE VECTOR INDEX idx_vecs ON vecs(emb);",
        SearchMode::Exact,
    )
    .unwrap();
    let idx_ms = t_idx.elapsed().as_secs_f64() * 1000.0;
    println!("[Bench 3] HNSW index build (10000 x 128d): {:.2} ms", idx_ms);

    // 3. Section 3.3 Query workload: 500 unfiltered + 500 filtered (10% selectivity), K = 10
    let k = 10usize;
    let mut exact_unfiltered_ms = 0.0;
    let mut ann_unfiltered_ms = 0.0;
    let mut unfiltered_hits = 0usize;

    let mut exact_filtered_ms = 0.0;
    let mut ann_filtered_ms = 0.0;
    let mut filtered_hits = 0usize;

    let total_per_group = 500usize;
    for q in 0..total_per_group {
        let qv: Vec<String> = (0..dim)
            .map(|d| format!("{:.4}", ((q * 13 + d) as f32 * 0.17).sin()))
            .collect();
        let q_literal = qv.join(",");

        let sql_u = format!(
            "SELECT id FROM vecs ORDER BY L2_DISTANCE(emb, '[{}]') LIMIT {};",
            q_literal, k
        );
        let te = Instant::now();
        let ExecResult::Query { rows: erows, .. } =
            db.execute_sql(&sql_u, SearchMode::Exact).unwrap()
        else {
            panic!();
        };
        exact_unfiltered_ms += te.elapsed().as_secs_f64() * 1000.0;

        let ta = Instant::now();
        let ExecResult::Query { rows: arows, .. } =
            db.execute_sql(&sql_u, SearchMode::Ann).unwrap()
        else {
            panic!();
        };
        ann_unfiltered_ms += ta.elapsed().as_secs_f64() * 1000.0;

        let eset: HashSet<i64> = erows.iter().map(|r| r[0].as_i64().unwrap()).collect();
        let aset: HashSet<i64> = arows.iter().map(|r| r[0].as_i64().unwrap()).collect();
        assert_eq!(aset.len(), k);
        unfiltered_hits += eset.intersection(&aset).count();

        let sql_f = format!(
            "SELECT id FROM vecs WHERE cat = {} ORDER BY L2_DISTANCE(emb, '[{}]') LIMIT {};",
            q % 10,
            q_literal,
            k
        );
        let tef = Instant::now();
        let ExecResult::Query { rows: efrows, .. } =
            db.execute_sql(&sql_f, SearchMode::Exact).unwrap()
        else {
            panic!();
        };
        exact_filtered_ms += tef.elapsed().as_secs_f64() * 1000.0;

        let taf = Instant::now();
        let ExecResult::Query { rows: afrows, .. } =
            db.execute_sql(&sql_f, SearchMode::Ann).unwrap()
        else {
            panic!();
        };
        ann_filtered_ms += taf.elapsed().as_secs_f64() * 1000.0;

        let efset: HashSet<i64> = efrows.iter().map(|r| r[0].as_i64().unwrap()).collect();
        let afset: HashSet<i64> = afrows.iter().map(|r| r[0].as_i64().unwrap()).collect();
        assert_eq!(afset.len(), k);
        filtered_hits += efset.intersection(&afset).count();
    }

    let u_recall = (unfiltered_hits as f64) / ((total_per_group * k) as f64);
    let f_recall = (filtered_hits as f64) / ((total_per_group * k) as f64);
    println!(
        "[Bench 4a Unfiltered x500] Exact: {:.2} ms ({:.3} ms/q) | ANN: {:.2} ms ({:.3} ms/q) | Recall@10: {:.4}",
        exact_unfiltered_ms,
        exact_unfiltered_ms / total_per_group as f64,
        ann_unfiltered_ms,
        ann_unfiltered_ms / total_per_group as f64,
        u_recall
    );
    println!(
        "[Bench 4b Filtered(10%) x500] Exact: {:.2} ms ({:.3} ms/q) | ANN: {:.2} ms ({:.3} ms/q) | Recall@10: {:.4}",
        exact_filtered_ms,
        exact_filtered_ms / total_per_group as f64,
        ann_filtered_ms,
        ann_filtered_ms / total_per_group as f64,
        f_recall
    );

    db.shutdown().unwrap();
    let _ = std::fs::remove_dir_all(&bench_dir);
}
