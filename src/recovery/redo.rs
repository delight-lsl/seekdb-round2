use crate::common::error::Result;
use crate::common::types::TxnId;
use crate::recovery::wal::{list_wal_segments, scan_wal_segment, WalFrameKind, WalManager};
use crate::storage::engine::StorageEngine;
use crate::transaction::operation::WalOp;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

/// Idempotent crash recovery executed on startup before emitting `ready`.
///
/// 1. Load `checkpoint.meta` (table schemas, generations, checkpointed file lengths).
/// 2. Scan the WAL after the checkpoint, cutting off a torn tail; collect the committed
///    transaction ids and their schema operations.
/// 3. Open every table file and rebuild its in-memory state from the complete groups:
///    self-committed groups always count, WAL-coordinated groups count only if they are
///    covered by the checkpoint or their transaction has a WAL commit record. The first
///    group that does not count (and anything after it) is truncated.
/// 4. Remove table files that belong to no committed table.
///
/// Nothing here depends on a previous recovery attempt having finished, so being killed
/// half-way and starting over yields the same state.
pub fn recover_database(data_dir: &Path) -> Result<(StorageEngine, WalManager)> {
    let (mut storage, ckp_opt) = StorageEngine::open_dirs(data_dir)?;
    let (start_wal_seq, start_wal_offset, ckp_next_txn_id) = match &ckp_opt {
        Some(ckp) => (ckp.wal_seq.max(1), ckp.wal_offset, ckp.next_txn_id.max(1)),
        None => (1, 0, 1),
    };

    let wal_dir = data_dir.join("wal");
    let segments = list_wal_segments(&wal_dir)?;

    let mut txn_ops: HashMap<TxnId, Vec<WalOp>> = HashMap::new();
    let mut committed_order: Vec<TxnId> = Vec::new();
    let mut committed: HashSet<TxnId> = HashSet::new();
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
                    if committed.insert(frame.txn_id) {
                        committed_order.push(frame.txn_id);
                    }
                }
            }
        }
        active_wal_seq = *seq;
        active_wal_offset = scan.valid_end_offset;
    }

    // Tables known to the checkpoint.
    if let Some(ckp) = &ckp_opt {
        storage.catalog.bump_next_table_id(ckp.next_table_id);
        for snap in &ckp.tables {
            storage.attach_table(
                snap.schema.clone(),
                snap.generation,
                snap.next_row_id,
                snap.safe_data_len,
                &committed,
            )?;
        }
    }

    // Schema operations committed after the checkpoint, in commit order.
    for txn_id in &committed_order {
        let Some(ops) = txn_ops.remove(txn_id) else {
            continue;
        };
        for op in ops {
            match op {
                WalOp::CreateTable(schema) => {
                    if storage.tables.contains_key(&schema.table_id) {
                        continue;
                    }
                    storage.attach_table(schema, 1, 1, 0, &committed)?;
                }
                WalOp::CreateIndex { table_id, index } => {
                    if storage
                        .catalog
                        .add_vector_index(table_id, index.clone())
                        .is_ok()
                    {
                        storage.get_table_mut(table_id)?.schema.indexes.push(index);
                    }
                }
            }
        }
    }

    // Drop table files that no committed table owns (aborted DDL, superseded generations).
    let expected: HashSet<String> = storage
        .tables
        .values()
        .filter_map(|dt| dt.file.path.file_name().and_then(|s| s.to_str()).map(String::from))
        .collect();
    if let Ok(entries) = fs::read_dir(data_dir.join("tables")) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if !expected.contains(name) {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
    }
    storage.enforce_memory_budget()?;

    let next_txn_id = ckp_next_txn_id.max(max_seen_txn_id + 1);
    let wal = WalManager::open(wal_dir, active_wal_seq, active_wal_offset, next_txn_id)?;
    wal.delete_segments_before(start_wal_seq)?;

    Ok((storage, wal))
}
