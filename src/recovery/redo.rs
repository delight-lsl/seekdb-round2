use crate::common::error::Result;
use crate::common::types::{RowId, TableId, TxnId};
use crate::recovery::checkpoint::{
    build_hnsw_from_disk, hnsw_snap_path, hnsw_spill_path, perform_checkpoint,
};
use crate::recovery::wal::{list_wal_segments, scan_wal_segment, WalFrameKind, WalManager};
use crate::storage::codec::decode_row;
use crate::storage::engine::StorageEngine;
use crate::transaction::operation::WalOp;
use crate::vector::index::HnswIndex;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Idempotent Redo-only crash recovery executed on startup before emitting `ready`.
pub fn recover_database(
    data_dir: &Path,
) -> Result<(StorageEngine, WalManager, HashMap<(TableId, usize), HnswIndex>)> {
    // Step 1 & 2: Load checkpoint.meta, truncate .dat files to safe lengths, rebuild PK & RowId maps
    let (mut storage, ckp_opt) = StorageEngine::open_from_checkpoint(data_dir)?;
    let (start_wal_seq, start_wal_offset, ckp_next_txn_id) = match &ckp_opt {
        Some(ckp) => (ckp.wal_seq.max(1), ckp.wal_offset, ckp.next_txn_id.max(1)),
        None => (1, 0, 1),
    };

    // Step 3: Scan WAL segments from checkpoint point onward, truncating any torn tail
    let wal_dir = data_dir.join("wal");
    let segments = list_wal_segments(&wal_dir)?;

    let mut txn_ops: HashMap<TxnId, Vec<WalOp>> = HashMap::new();
    let mut committed_order: Vec<TxnId> = Vec::new();
    let mut committed_set: HashSet<TxnId> = HashSet::new();
    let mut max_seen_txn_id = 0_u64;
    let mut active_wal_seq = start_wal_seq;
    let mut active_wal_offset = start_wal_offset;

    for (seq, path) in &segments {
        if *seq < start_wal_seq {
            continue;
        }
        let off = if *seq == start_wal_seq {
            start_wal_offset
        } else {
            0
        };
        let scan = scan_wal_segment(path, off)?;
        for frame in scan.frames {
            max_seen_txn_id = max_seen_txn_id.max(frame.txn_id);
            match frame.kind {
                WalFrameKind::Begin => {
                    txn_ops.entry(frame.txn_id).or_default();
                }
                WalFrameKind::Op(op) => {
                    txn_ops.entry(frame.txn_id).or_default().push(op);
                }
                WalFrameKind::Commit => {
                    if committed_set.insert(frame.txn_id) {
                        committed_order.push(frame.txn_id);
                    }
                }
            }
        }
        active_wal_seq = *seq;
        active_wal_offset = scan.valid_end_offset;
    }

    // Step 4: Replay ONLY committed transactions in commit order
    let mut replayed_any = false;
    let mut touched_tables: HashSet<TableId> = HashSet::new();

    for txn_id in &committed_order {
        let Some(ops) = txn_ops.remove(txn_id) else {
            continue;
        };
        replayed_any = true;
        // Group consecutive table DML ops per table for batched apply
        let mut pending_puts: HashMap<TableId, Vec<(RowId, Option<i64>, crate::common::types::Row)>> =
            HashMap::new();
        let mut pending_deletes: HashMap<TableId, Vec<(RowId, Option<i64>)>> = HashMap::new();

        for op in ops {
            match op {
                WalOp::CreateTable(schema) => {
                    let tid = schema.table_id;
                    storage.create_table(schema)?;
                    touched_tables.insert(tid);
                }
                WalOp::CreateIndex { table_id, index } => {
                    storage
                        .catalog
                        .add_vector_index(table_id, index.clone())?;
                    storage
                        .get_table_mut(table_id)?
                        .schema
                        .indexes
                        .push(index);
                    touched_tables.insert(table_id);
                }
                WalOp::PutRow {
                    table_id,
                    row_id,
                    old_pk,
                    row_bytes,
                } => {
                    let schema = &storage.get_table(table_id)?.schema;
                    let row = decode_row(&row_bytes, schema)?;
                    pending_puts
                        .entry(table_id)
                        .or_default()
                        .push((row_id, old_pk, row));
                    touched_tables.insert(table_id);
                }
                WalOp::DeleteRow {
                    table_id,
                    row_id,
                    old_pk,
                } => {
                    pending_deletes
                        .entry(table_id)
                        .or_default()
                        .push((row_id, old_pk));
                    touched_tables.insert(table_id);
                }
            }
        }

        for tid in pending_puts.keys().chain(pending_deletes.keys()).copied().collect::<HashSet<_>>() {
            let puts = pending_puts.remove(&tid).unwrap_or_default();
            let dels = pending_deletes.remove(&tid).unwrap_or_default();
            let dt = storage.get_table_mut(tid)?;
            dt.apply_batch(&puts, &dels, false)?;
        }
    }

    for &tid in &touched_tables {
        if let Ok(dt) = storage.get_table(tid) {
            dt.file.sync_all()?;
        }
    }

    // Step 5: Open WalManager at latest valid WAL end
    let next_txn_id = ckp_next_txn_id.max(max_seen_txn_id + 1);
    let mut wal = WalManager::open(wal_dir, active_wal_seq, active_wal_offset, next_txn_id)?;

    // Step 6: Load HNSW snapshots for untouched tables, or rebuild cleanly for touched/missing tables
    let mut hnsw_indexes: HashMap<(TableId, usize), HnswIndex> = HashMap::new();
    let mut to_rebuild = Vec::new();
    for (&tid, dt) in &storage.tables {
        for idx_schema in &dt.schema.indexes {
            if !touched_tables.contains(&tid) {
                let snap_path =
                    hnsw_snap_path(data_dir, tid, idx_schema.col_idx, dt.generation);
                let spill_path =
                    hnsw_spill_path(data_dir, tid, idx_schema.col_idx, dt.generation);
                if let Some(loaded) =
                    HnswIndex::load_snapshot(idx_schema.name.clone(), &snap_path, spill_path)?
                {
                    if loaded.live_count() == dt.live_count() && loaded.dim == idx_schema.dim {
                        hnsw_indexes.insert((tid, idx_schema.col_idx), loaded);
                        continue;
                    }
                }
            }
            to_rebuild.push((tid, idx_schema.clone()));
        }
    }
    for (tid, idx_schema) in to_rebuild {
        let dt = storage.get_table(tid)?;
        let hnsw = build_hnsw_from_disk(
            data_dir,
            dt,
            idx_schema.col_idx,
            idx_schema.dim,
            &idx_schema.name,
        )?;
        hnsw_indexes.insert((tid, idx_schema.col_idx), hnsw);
    }

    // Step 7: Persist clean checkpoint if WAL Redo replayed any transactions
    if replayed_any {
        perform_checkpoint(&mut storage, &mut wal, &mut hnsw_indexes, false)?;
    } else {
        wal.delete_segments_before(start_wal_seq)?;
    }

    Ok((storage, wal, hnsw_indexes))
}
