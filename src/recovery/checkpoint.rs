use crate::common::error::Result;
use crate::common::types::{TableId, Value};
use crate::recovery::wal::WalManager;
use crate::storage::engine::{pk_of_row, DiskTable, StorageEngine};
use crate::storage::file::sync_dir;
use crate::vector::index::HnswIndex;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub fn hnsw_snap_path(data_dir: &Path, tid: TableId, col_idx: usize, generation: u64) -> PathBuf {
    data_dir
        .join("tables")
        .join(format!("idx_{}_{}_g{}.hnsw", tid, col_idx, generation))
}

pub fn hnsw_spill_path(data_dir: &Path, tid: TableId, col_idx: usize, generation: u64) -> PathBuf {
    data_dir
        .join("tables")
        .join(format!("idx_{}_{}_g{}.vec", tid, col_idx, generation))
}

pub fn build_hnsw_from_disk(
    data_dir: &Path,
    dt: &DiskTable,
    col_idx: usize,
    dim: usize,
    index_name: &str,
) -> Result<HnswIndex> {
    let spill_path = hnsw_spill_path(data_dir, dt.schema.table_id, col_idx, dt.generation);
    let mut hnsw = HnswIndex::new(index_name.to_string(), dim, spill_path);
    let mut needed = vec![false; dt.schema.columns.len()];
    needed[col_idx] = true;
    if let Some(pk_idx) = dt.schema.pk_col_idx {
        needed[pk_idx] = true;
    }
    dt.scan_live_streaming(&needed, |rid, _payload, row| {
        let pk = pk_of_row(&dt.schema, &row);
        if let Value::Vector(ref vec) = row.values[col_idx] {
            hnsw.upsert(rid, pk, vec)?;
        }
        Ok(true)
    })?;
    Ok(hnsw)
}

/// Atomically persist checkpoint state, compact bloated tables, rotate WAL, and clean up old segments.
pub fn perform_checkpoint(
    storage: &mut StorageEngine,
    wal: &mut WalManager,
    hnsw_indexes: &mut HashMap<(TableId, usize), HnswIndex>,
    force_compact: bool,
) -> Result<()> {
    let data_dir = storage.data_dir.clone();
    let mut old_files_to_delete: Vec<PathBuf> = Vec::new();

    // 1. Compact bloated tables into new generation files
    let table_ids: Vec<TableId> = storage.tables.keys().copied().collect();
    for tid in table_ids {
        let should_compact = {
            let dt = storage.get_table(tid)?;
            force_compact || dt.needs_compaction()
        };
        if should_compact {
            let old_gen = storage.get_table(tid)?.generation;
            let old_dat = {
                let dt = storage.get_table_mut(tid)?;
                dt.compact_to_next_generation(&data_dir)?
            };
            old_files_to_delete.push(old_dat);

            // Rebuild HNSW snapshots at the new generation so old .hnsw/.vec can be cleaned up
            let dt = storage.get_table(tid)?;
            for idx_schema in &dt.schema.indexes {
                old_files_to_delete.push(hnsw_snap_path(&data_dir, tid, idx_schema.col_idx, old_gen));
                old_files_to_delete.push(hnsw_spill_path(&data_dir, tid, idx_schema.col_idx, old_gen));
                let rebuilt = build_hnsw_from_disk(
                    &data_dir,
                    dt,
                    idx_schema.col_idx,
                    idx_schema.dim,
                    &idx_schema.name,
                )?;
                hnsw_indexes.insert((tid, idx_schema.col_idx), rebuilt);
            }
        }
    }

    // 2. Sync all table data files & save HNSW snapshots
    for dt in storage.tables.values() {
        dt.file.sync_all()?;
    }
    for ((tid, col_idx), hnsw) in hnsw_indexes.iter_mut() {
        if let Ok(dt) = storage.get_table(*tid) {
            let snap = hnsw_snap_path(&data_dir, *tid, *col_idx, dt.generation);
            let tmp = data_dir
                .join("tmp")
                .join(format!("idx_{}_{}_g{}.hnsw.tmp", tid, col_idx, dt.generation));
            hnsw.save_snapshot(&snap, &tmp)?;
        }
    }

    // 3. Rotate WAL to a clean segment starting at offset 0
    let (new_wal_seq, new_wal_offset) = wal.rotate_segment()?;

    // 4. Atomically replace checkpoint.meta
    storage.save_checkpoint_snapshot(new_wal_seq, new_wal_offset, wal.next_txn_id)?;

    // 5. Delete superseded WAL segments and old generation data/index files
    wal.delete_segments_before(new_wal_seq)?;
    let tables_dir = data_dir.join("tables");
    let mut deleted_table_files = false;
    for path in old_files_to_delete {
        if path.exists() && fs::remove_file(&path).is_ok() {
            deleted_table_files = true;
        }
    }
    if deleted_table_files && tables_dir.exists() {
        let _ = sync_dir(&tables_dir);
    }

    Ok(())
}
