use crate::common::error::Result;
use crate::common::types::TableId;
use crate::recovery::wal::WalManager;
use crate::storage::engine::StorageEngine;
use crate::storage::file::sync_dir;
use std::fs;
use std::path::PathBuf;

/// Atomically persist checkpoint state, compact bloated tables, rotate WAL, and clean up old segments.
pub fn perform_checkpoint(
    storage: &mut StorageEngine,
    wal: &mut WalManager,
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
            let dt = storage.get_table_mut(tid)?;
            old_files_to_delete.push(dt.compact_to_next_generation(&data_dir)?);
        }
    }

    // 2. Rotate WAL to a clean segment starting at offset 0
    let (new_wal_seq, new_wal_offset) = wal.rotate_segment()?;

    // 3. Atomically replace checkpoint.meta (table files are synced inside)
    storage.save_checkpoint_snapshot(new_wal_seq, new_wal_offset, wal.next_txn_id)?;

    // 4. Delete superseded WAL segments and old generation data files
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
