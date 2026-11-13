use crate::catalog::manager::Catalog;
use crate::catalog::schema::TableSchema;
use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::{Cell, Row, RowAccess, RowId, TableId, TxnId, Value};
use crate::storage::codec::{
    decode_row_selective, decode_schema, encode_row, encode_schema, extract_pk_from_row_bytes,
    write_u32, write_u64, Cursor,
};
use crate::storage::file::{atomic_write_file, sync_dir, TableDataFile};
use crate::storage::mem::{MemTable, SlotRef};
use crate::storage::record::{begin_data_frame, finish_data_frame, DataRecordOp};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::{Path, PathBuf};

/// Total heap budget for in-memory table images; tables beyond it are served from disk.
///
/// A quarter of the container memory limit (cgroup v2 `memory.max`, cgroup v1
/// `memory.limit_in_bytes`, bounded by physical RAM), capped at 512 MiB. The remaining
/// three quarters cover per-statement working memory, lookup maps and the page cache.
/// `MVDB_MEM_BUDGET_MB` overrides the computed value.
pub fn mem_budget_bytes() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        if let Some(mb) = std::env::var("MVDB_MEM_BUDGET_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            return mb.saturating_mul(1024 * 1024);
        }
        let read_num = |path: &str| -> Option<u64> {
            fs::read_to_string(path).ok()?.trim().parse::<u64>().ok()
        };
        let mut limit: u64 = u64::MAX;
        for path in [
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ] {
            if let Some(v) = read_num(path) {
                limit = limit.min(v);
            }
        }
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            if let Some(kb) = meminfo
                .lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
            {
                limit = limit.min(kb.saturating_mul(1024));
            }
        }
        const CAP: u64 = 512 * 1024 * 1024;
        (limit / 4).min(CAP) as usize
    })
}

/// Cheap hasher for integer keys (row ids, primary keys).
#[derive(Default, Clone, Copy)]
pub struct IdHasher(u64);

impl Hasher for IdHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    #[inline]
    fn write_u64(&mut self, x: u64) {
        self.0 = (x ^ (x >> 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    #[inline]
    fn write_i64(&mut self, x: i64) {
        self.write_u64(x as u64);
    }
    #[inline]
    fn write_u32(&mut self, x: u32) {
        self.write_u64(x as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 29)
    }
}

pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<IdHasher>>;

/// How a transaction's frame group in a table file is terminated.
#[derive(Debug, Clone, Copy)]
pub enum GroupEnd {
    /// Self-contained commit: durable as soon as the table file is synced.
    Commit,
    /// Part of a transaction whose commit point is a record in the WAL.
    Pending(TxnId),
}

/// Where the live rows of a table are served from.
#[derive(Debug)]
pub enum RowStore {
    /// Whole table in the columnar memory image; `slots` maps row id -> slot.
    Mem {
        mem: MemTable,
        slots: FastMap<RowId, u32>,
    },
    /// Table larger than the memory budget: only frame locations are kept on heap.
    Disk {
        locations: FastMap<RowId, (u64, u32)>,
    },
}

/// A live row handed to scan callbacks.
pub enum LiveRow<'a> {
    Mem(SlotRef<'a>, &'a MemTable, u32),
    Disk {
        partial: &'a Row,
        payload: &'a [u8],
        schema: &'a TableSchema,
    },
}

impl RowAccess for LiveRow<'_> {
    #[inline]
    fn cell(&self, idx: usize) -> Cell<'_> {
        match self {
            LiveRow::Mem(r, _, _) => r.cell(idx),
            LiveRow::Disk { partial, .. } => partial.cell(idx),
        }
    }
}

impl LiveRow<'_> {
    /// Materialize the columns flagged in `needed` (others become NULL placeholders).
    pub fn materialize(&self, needed: &[bool]) -> Result<Row> {
        match self {
            LiveRow::Mem(_, mem, slot) => Ok(mem.materialize(*slot, Some(needed))),
            LiveRow::Disk {
                payload, schema, ..
            } => decode_row_selective(payload, schema, needed),
        }
    }
}

/// Storage for a single table: an append-only commit-log data file plus either a
/// full in-memory columnar image or (beyond the memory budget) a location index.
#[derive(Debug)]
pub struct DiskTable {
    pub schema: TableSchema,
    pub generation: u64,
    pub next_row_id: RowId,
    pub file: TableDataFile,
    pub pk_index: FastMap<i64, RowId>,
    pub store: RowStore,
    /// Row frames (puts + tombstones) currently in the file.
    pub total_frames: u64,
    scratch: Vec<u8>,
}

impl DiskTable {
    pub fn table_file_path(data_dir: &Path, table_id: TableId, generation: u64) -> PathBuf {
        data_dir
            .join("tables")
            .join(format!("t_{}_g{}.dat", table_id, generation))
    }

    /// Open (or create) the table file and rebuild the in-memory state from it.
    ///
    /// Only complete groups count: a group ended by `Commit` is always applied; one
    /// ended by `Pending(txn)` is applied when it lies below `safe_len` (covered by a
    /// checkpoint) or `txn` has a commit record in the WAL. The first group that does
    /// not qualify, or any torn/corrupt tail, is cut off.
    pub fn open_or_create(
        data_dir: &Path,
        schema: TableSchema,
        generation: u64,
        next_row_id: RowId,
        safe_len: u64,
        committed: &HashSet<TxnId>,
        mem_budget_left: usize,
    ) -> Result<Self> {
        let path = Self::table_file_path(data_dir, schema.table_id, generation);
        let mut file = TableDataFile::open(path)?;
        if file.created {
            if let Some(parent) = file.path.parent() {
                sync_dir(parent)?;
            }
        }
        let scan_len = file.len();
        let mem_mode = scan_len as usize <= mem_budget_left;

        let mut pk_index: FastMap<i64, RowId> = FastMap::default();
        let mut mem = MemTable::new(&schema);
        let mut slots: FastMap<RowId, u32> = FastMap::default();
        let mut locations: FastMap<RowId, (u64, u32)> = FastMap::default();
        let mut disk_pks: FastMap<RowId, i64> = FastMap::default();
        let mut max_row_id = 0_u64;
        let mut total_frames = 0_u64;
        let mut valid_len = 0_u64;
        let pk_col = schema.pk_col_idx;

        // Frames of the group currently being read: (op, row id, payload range | location).
        let mut group: Vec<(DataRecordOp, RowId, u64, u32)> = Vec::new();
        let mut gbuf: Vec<u8> = Vec::new();

        file.scan_raw(scan_len, true, mem_mode, |op, rid, payload, off, flen| {
            match op {
                DataRecordOp::Put | DataRecordOp::Tombstone => {
                    if mem_mode {
                        group.push((op, rid, gbuf.len() as u64, payload.len() as u32));
                        gbuf.extend_from_slice(payload);
                    } else {
                        if op == DataRecordOp::Put && pk_col.is_some() {
                            let pk = extract_pk_from_row_bytes(payload, pk_col.unwrap())?;
                            gbuf.extend_from_slice(&pk.unwrap_or(0).to_le_bytes());
                        }
                        group.push((op, rid, off, flen));
                    }
                }
                DataRecordOp::Commit | DataRecordOp::Pending => {
                    let ok = op == DataRecordOp::Commit || off < safe_len || committed.contains(&rid);
                    if !ok {
                        return Ok(false);
                    }
                    let mut pk_cursor = 0usize;
                    for &(gop, grid, a, b) in &group {
                        total_frames += 1;
                        max_row_id = max_row_id.max(grid);
                        if mem_mode {
                            match gop {
                                DataRecordOp::Put => {
                                    let bytes = &gbuf[a as usize..a as usize + b as usize];
                                    let slot = match slots.get(&grid) {
                                        Some(&s) => {
                                            if let Some(old_pk) = mem.pk(s) {
                                                if pk_index.get(&old_pk) == Some(&grid) {
                                                    pk_index.remove(&old_pk);
                                                }
                                            }
                                            s
                                        }
                                        None => {
                                            let s = mem.alloc_slot();
                                            slots.insert(grid, s);
                                            s
                                        }
                                    };
                                    mem.set_from_bytes(slot, grid, bytes)?;
                                    if let Some(pk) = mem.pk(slot) {
                                        pk_index.insert(pk, grid);
                                    }
                                }
                                _ => {
                                    if let Some(s) = slots.remove(&grid) {
                                        if let Some(old_pk) = mem.pk(s) {
                                            if pk_index.get(&old_pk) == Some(&grid) {
                                                pk_index.remove(&old_pk);
                                            }
                                        }
                                        mem.free_slot(s);
                                    }
                                }
                            }
                        } else {
                            if let Some(old_pk) = disk_pks.remove(&grid) {
                                if pk_index.get(&old_pk) == Some(&grid) {
                                    pk_index.remove(&old_pk);
                                }
                            }
                            match gop {
                                DataRecordOp::Put => {
                                    if pk_col.is_some() {
                                        let mut raw = [0u8; 8];
                                        raw.copy_from_slice(&gbuf[pk_cursor..pk_cursor + 8]);
                                        pk_cursor += 8;
                                        let pk = i64::from_le_bytes(raw);
                                        pk_index.insert(pk, grid);
                                        disk_pks.insert(grid, pk);
                                    }
                                    locations.insert(grid, (a, b));
                                }
                                _ => {
                                    locations.remove(&grid);
                                }
                            }
                        }
                    }
                    group.clear();
                    gbuf.clear();
                    valid_len = off + flen as u64;
                }
            }
            Ok(true)
        })?;
        drop(group);
        drop(gbuf);
        drop(disk_pks);

        if valid_len != scan_len {
            file.truncate(valid_len)?;
        }
        let store = if mem_mode {
            file.drop_page_cache();
            mem.maybe_compact();
            RowStore::Mem { mem, slots }
        } else {
            RowStore::Disk { locations }
        };

        Ok(Self {
            schema,
            generation,
            next_row_id: next_row_id.max(max_row_id + 1),
            file,
            pk_index,
            store,
            total_frames,
            scratch: Vec::new(),
        })
    }

    #[inline]
    pub fn is_mem(&self) -> bool {
        matches!(self.store, RowStore::Mem { .. })
    }

    #[inline]
    pub fn mem(&self) -> Option<&MemTable> {
        match &self.store {
            RowStore::Mem { mem, .. } => Some(mem),
            RowStore::Disk { .. } => None,
        }
    }

    pub fn mem_bytes(&self) -> usize {
        self.mem().map_or(0, |m| m.bytes())
    }

    pub fn live_count(&self) -> usize {
        match &self.store {
            RowStore::Mem { mem, .. } => mem.live(),
            RowStore::Disk { locations } => locations.len(),
        }
    }

    pub fn contains_row(&self, row_id: RowId) -> bool {
        match &self.store {
            RowStore::Mem { slots, .. } => slots.contains_key(&row_id),
            RowStore::Disk { locations } => locations.contains_key(&row_id),
        }
    }

    /// Prepare the in-memory image for vector search on `col`.
    pub fn prepare_vector_search(&mut self, col: usize) {
        if let RowStore::Mem { mem, .. } = &mut self.store {
            mem.prepare_vector_search(col);
        }
    }

    #[inline]
    pub fn slot_of(&self, row_id: RowId) -> Option<u32> {
        match &self.store {
            RowStore::Mem { slots, .. } => slots.get(&row_id).copied(),
            RowStore::Disk { .. } => None,
        }
    }

    /// Run `f` on the live row `row_id` (if it exists) with `access` columns readable.
    pub fn with_row<T, F>(&self, row_id: RowId, access: &[bool], f: F) -> Result<Option<T>>
    where
        F: FnOnce(LiveRow<'_>) -> Result<T>,
    {
        match &self.store {
            RowStore::Mem { mem, slots } => match slots.get(&row_id) {
                Some(&slot) => Ok(Some(f(LiveRow::Mem(mem.row(slot), mem, slot))?)),
                None => Ok(None),
            },
            RowStore::Disk { locations } => match locations.get(&row_id) {
                Some(&(off, len)) => {
                    let frame = self.file.read_frame_at(off, len)?;
                    let partial = decode_row_selective(&frame.payload, &self.schema, access)?;
                    Ok(Some(f(LiveRow::Disk {
                        partial: &partial,
                        payload: &frame.payload,
                        schema: &self.schema,
                    })?))
                }
                None => Ok(None),
            },
        }
    }

    /// Visit every live row; `access` flags the columns the callback reads through `cell()`.
    /// The callback returns `false` to stop early.
    pub fn scan_live<F>(&self, access: &[bool], mut on_row: F) -> Result<()>
    where
        F: FnMut(RowId, LiveRow<'_>) -> Result<bool>,
    {
        match &self.store {
            RowStore::Mem { mem, .. } => {
                for slot in 0..mem.nslots() {
                    let rid = mem.rid(slot);
                    if rid == 0 {
                        continue;
                    }
                    let s = slot as u32;
                    if !on_row(rid, LiveRow::Mem(mem.row(s), mem, s))? {
                        break;
                    }
                }
                Ok(())
            }
            RowStore::Disk { locations } => {
                let scan_len = self.file.len();
                let schema = &self.schema;
                self.file
                    .scan_raw(scan_len, false, false, |op, rid, payload, off, flen| {
                        if op != DataRecordOp::Put || locations.get(&rid) != Some(&(off, flen)) {
                            return Ok(true);
                        }
                        let partial = decode_row_selective(payload, schema, access)?;
                        on_row(
                            rid,
                            LiveRow::Disk {
                                partial: &partial,
                                payload,
                                schema,
                            },
                        )
                    })?;
                Ok(())
            }
        }
    }

    /// Append one transaction group (row frames + terminator) and make it durable.
    /// Nothing is visible in memory until `apply_group` is called with the result.
    pub fn append_group(
        &mut self,
        puts: &[(RowId, Option<i64>, Row)],
        deletes: &[(RowId, Option<i64>)],
        end: GroupEnd,
    ) -> Result<Vec<(u64, u32)>> {
        let disk = !self.is_mem();
        let mut buf = std::mem::take(&mut self.scratch);
        buf.clear();
        let base = self.file.len();
        let mut locs = Vec::with_capacity(if disk { puts.len() } else { 0 });
        let mut res: Result<()> = Ok(());
        for (rid, _, row) in puts {
            let start = begin_data_frame(DataRecordOp::Put, *rid, &mut buf);
            if let Err(e) = encode_row(row, &mut buf) {
                res = Err(e);
                break;
            }
            match finish_data_frame(start, &mut buf) {
                Ok(flen) => {
                    if disk {
                        locs.push((base + start as u64, flen));
                    }
                }
                Err(e) => {
                    res = Err(e);
                    break;
                }
            }
        }
        if res.is_ok() {
            for (rid, _) in deletes {
                let start = begin_data_frame(DataRecordOp::Tombstone, *rid, &mut buf);
                if let Err(e) = finish_data_frame(start, &mut buf) {
                    res = Err(e);
                    break;
                }
            }
        }
        if res.is_ok() {
            let (op, id) = match end {
                GroupEnd::Commit => (DataRecordOp::Commit, 0),
                GroupEnd::Pending(t) => (DataRecordOp::Pending, t),
            };
            let start = begin_data_frame(op, id, &mut buf);
            // Large groups end on a page boundary (zero padding inside the terminator), so
            // the tail page is not physically written again by the next commit.
            if buf.len() >= 16 * 1024 {
                let end = base as usize + buf.len() + 4;
                let pad = (4096 - end % 4096) % 4096;
                buf.resize(buf.len() + pad, 0);
            }
            res = finish_data_frame(start, &mut buf).map(|_| ());
        }
        if res.is_ok() {
            res = self.file.commit_append(&buf, !disk).map(|_| ());
        }
        if buf.capacity() <= 4 * 1024 * 1024 {
            self.scratch = buf;
        }
        res?;
        self.total_frames += (puts.len() + deletes.len()) as u64;
        Ok(locs)
    }

    /// Cut the file back to `len` (used to undo a group of a transaction that failed
    /// before reaching its commit point).
    pub fn rollback_file_to(&mut self, len: u64, frames: u64) -> Result<()> {
        self.file.truncate(len)?;
        self.total_frames = frames;
        Ok(())
    }

    /// Publish a durable group to the in-memory state.
    pub fn apply_group(
        &mut self,
        puts: Vec<(RowId, Option<i64>, Row)>,
        deletes: &[(RowId, Option<i64>)],
        locs: Vec<(u64, u32)>,
    ) -> Result<()> {
        for (rid, old_pk, _) in &puts {
            if let Some(opk) = old_pk {
                if self.pk_index.get(opk) == Some(rid) {
                    self.pk_index.remove(opk);
                }
            }
        }
        for (rid, old_pk) in deletes {
            if let Some(opk) = old_pk {
                if self.pk_index.get(opk) == Some(rid) {
                    self.pk_index.remove(opk);
                }
            }
        }
        let pk_col = self.schema.pk_col_idx;
        let mut max_rid = 0;
        match &mut self.store {
            RowStore::Mem { mem, slots } => {
                for (rid, _) in deletes {
                    if let Some(slot) = slots.remove(rid) {
                        mem.free_slot(slot);
                    }
                }
                let churn = !deletes.is_empty();
                let mut updated = false;
                for (rid, _, row) in &puts {
                    let slot = match slots.get(rid) {
                        Some(&s) => {
                            updated = true;
                            s
                        }
                        None => {
                            let s = mem.alloc_slot();
                            slots.insert(*rid, s);
                            s
                        }
                    };
                    mem.set_row(slot, *rid, row)?;
                    max_rid = max_rid.max(*rid);
                    if let Some(pi) = pk_col {
                        if let Value::BigInt(pk) = row.values[pi] {
                            self.pk_index.insert(pk, *rid);
                        }
                    }
                }
                if churn || updated {
                    mem.maybe_compact();
                }
            }
            RowStore::Disk { locations } => {
                for (rid, _) in deletes {
                    locations.remove(rid);
                }
                for ((rid, _, row), loc) in puts.iter().zip(locs.into_iter()) {
                    locations.insert(*rid, loc);
                    max_rid = max_rid.max(*rid);
                    if let Some(pi) = pk_col {
                        if let Value::BigInt(pk) = row.values[pi] {
                            self.pk_index.insert(pk, *rid);
                        }
                    }
                }
            }
        }
        self.next_row_id = self.next_row_id.max(max_rid + 1);
        Ok(())
    }

    /// Give up the in-memory image and serve this table from its file.
    pub fn spill_to_disk(&mut self) -> Result<()> {
        if !self.is_mem() {
            return Ok(());
        }
        let mut locations: FastMap<RowId, (u64, u32)> = FastMap::default();
        let scan_len = self.file.len();
        self.file
            .scan_raw(scan_len, false, false, |op, rid, _payload, off, flen| {
                match op {
                    DataRecordOp::Put => {
                        locations.insert(rid, (off, flen));
                    }
                    DataRecordOp::Tombstone => {
                        locations.remove(&rid);
                    }
                    _ => {}
                }
                Ok(true)
            })?;
        self.store = RowStore::Disk { locations };
        Ok(())
    }

    /// Needs compaction if dead frames are >= 50% and file > 8 MiB.
    pub fn needs_compaction(&self) -> bool {
        let live = self.live_count() as u64;
        self.file.len() > 8 * 1024 * 1024 && self.total_frames > live.saturating_mul(2) + 1024
    }

    /// Rewrite the live rows into a new generation file `t_<tid>_g<gen+1>.dat`.
    /// Returns the old file path to be deleted AFTER `checkpoint.meta` is updated.
    pub fn compact_to_next_generation(&mut self, data_dir: &Path) -> Result<PathBuf> {
        let old_path = self.file.path.clone();
        let next_gen = self.generation + 1;
        let new_path = Self::table_file_path(data_dir, self.schema.table_id, next_gen);
        let tmp_path = data_dir
            .join("tmp")
            .join(format!("compact_t_{}_g{}.dat.tmp", self.schema.table_id, next_gen));
        if let Some(p) = tmp_path.parent() {
            fs::create_dir_all(p)?;
        }
        let _ = fs::remove_file(&tmp_path);

        let mut out = TableDataFile::open(tmp_path.clone())?;
        let mut buf: Vec<u8> = Vec::with_capacity(512 * 1024);
        let mut new_locations: FastMap<RowId, (u64, u32)> = FastMap::default();
        let mut total_live = 0_u64;
        match &self.store {
            RowStore::Mem { mem, .. } => {
                for slot in 0..mem.nslots() {
                    let rid = mem.rid(slot);
                    if rid == 0 {
                        continue;
                    }
                    let row = mem.materialize(slot as u32, None);
                    let start = begin_data_frame(DataRecordOp::Put, rid, &mut buf);
                    encode_row(&row, &mut buf)?;
                    finish_data_frame(start, &mut buf)?;
                    total_live += 1;
                    if buf.len() >= 256 * 1024 {
                        out.append_unsynced(&buf)?;
                        buf.clear();
                    }
                }
            }
            RowStore::Disk { locations } => {
                let scan_len = self.file.len();
                let mut written = 0_u64;
                self.file
                    .scan_raw(scan_len, false, false, |op, rid, payload, off, flen| {
                        if op == DataRecordOp::Put && locations.get(&rid) == Some(&(off, flen)) {
                            let start = begin_data_frame(DataRecordOp::Put, rid, &mut buf);
                            buf.extend_from_slice(payload);
                            let nlen = finish_data_frame(start, &mut buf)?;
                            new_locations.insert(rid, (written + start as u64, nlen));
                            total_live += 1;
                            if buf.len() >= 256 * 1024 {
                                out.append_unsynced(&buf)?;
                                written += buf.len() as u64;
                                buf.clear();
                            }
                        }
                        Ok(true)
                    })?;
            }
        }
        let start = begin_data_frame(DataRecordOp::Commit, 0, &mut buf);
        finish_data_frame(start, &mut buf)?;
        out.append_unsynced(&buf)?;
        out.sync_all()?;
        drop(out);

        fs::rename(&tmp_path, &new_path)?;
        if let Some(p) = new_path.parent() {
            sync_dir(p)?;
        }

        self.file = TableDataFile::open(new_path)?;
        self.generation = next_gen;
        self.total_frames = total_live;
        match &mut self.store {
            RowStore::Mem { .. } => self.file.drop_page_cache(),
            RowStore::Disk { locations } => *locations = new_locations,
        }
        Ok(old_path)
    }
}

pub fn pk_of_row(schema: &TableSchema, row: &Row) -> Option<i64> {
    schema.pk_col_idx.and_then(|idx| match row.values.get(idx) {
        Some(Value::BigInt(v)) => Some(*v),
        _ => None,
    })
}

/// Snapshot metadata for a single table stored inside `checkpoint.meta`.
#[derive(Debug, Clone)]
pub struct TableMetaSnapshot {
    pub schema: TableSchema,
    pub generation: u64,
    pub safe_data_len: u64,
    pub next_row_id: RowId,
}

#[derive(Debug, Clone)]
pub struct CheckpointMeta {
    pub wal_seq: u64,
    pub wal_offset: u64,
    pub next_txn_id: u64,
    pub next_table_id: TableId,
    pub tables: Vec<TableMetaSnapshot>,
}

const CKP_MAGIC: [u8; 4] = *b"CKP2";
const CKP_VERSION: u32 = 2;

impl CheckpointMeta {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&CKP_MAGIC);
        write_u32(&mut buf, CKP_VERSION);
        write_u64(&mut buf, self.wal_seq);
        write_u64(&mut buf, self.wal_offset);
        write_u64(&mut buf, self.next_txn_id);
        write_u32(&mut buf, self.next_table_id);
        write_u32(&mut buf, self.tables.len() as u32);
        for t in &self.tables {
            encode_schema(&t.schema, &mut buf)?;
            write_u64(&mut buf, t.generation);
            write_u64(&mut buf, t.safe_data_len);
            write_u64(&mut buf, t.next_row_id);
        }
        let crc = crc32fast::hash(&buf);
        write_u32(&mut buf, crc);
        Ok(buf)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 + 4 + 8 + 8 + 8 + 4 + 4 + 4 {
            return Err(DbError::new(ErrorCode::IoError, "checkpoint file too short"));
        }
        let body_len = bytes.len() - 4;
        let expected_crc = u32::from_le_bytes(bytes[body_len..].try_into().unwrap());
        let actual_crc = crc32fast::hash(&bytes[..body_len]);
        if actual_crc != expected_crc {
            return Err(DbError::new(ErrorCode::IoError, "checkpoint CRC mismatch"));
        }
        let mut cur = Cursor::new(&bytes[..body_len]);
        if cur.read_bytes(4)? != CKP_MAGIC {
            return Err(DbError::new(ErrorCode::IoError, "bad checkpoint magic"));
        }
        let ver = cur.read_u32()?;
        if ver != CKP_VERSION {
            return Err(DbError::new(
                ErrorCode::IoError,
                format!("unsupported checkpoint version {}", ver),
            ));
        }
        let wal_seq = cur.read_u64()?;
        let wal_offset = cur.read_u64()?;
        let next_txn_id = cur.read_u64()?;
        let next_table_id = cur.read_u32()?;
        let tcount = cur.read_u32()? as usize;
        let mut tables = Vec::with_capacity(tcount);
        for _ in 0..tcount {
            let schema = decode_schema(&mut cur)?;
            let generation = cur.read_u64()?;
            let safe_data_len = cur.read_u64()?;
            let next_row_id = cur.read_u64()?;
            tables.push(TableMetaSnapshot {
                schema,
                generation,
                safe_data_len,
                next_row_id,
            });
        }
        if cur.remaining() != 0 {
            return Err(DbError::new(
                ErrorCode::IoError,
                "trailing bytes in checkpoint metadata",
            ));
        }
        Ok(Self {
            wal_seq,
            wal_offset,
            next_txn_id,
            next_table_id,
            tables,
        })
    }

    pub fn save_atomic(&self, data_dir: &Path) -> Result<()> {
        let bytes = self.encode()?;
        let target = data_dir.join("checkpoint.meta");
        let tmp = data_dir.join("tmp").join("checkpoint.meta.tmp");
        atomic_write_file(&target, &tmp, &bytes)
    }

    pub fn load_if_exists(data_dir: &Path) -> Result<Option<Self>> {
        let target = data_dir.join("checkpoint.meta");
        if !target.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&target)?;
        Ok(Some(Self::decode(&bytes)?))
    }
}

/// Manages table storage and catalog metadata.
#[derive(Debug)]
pub struct StorageEngine {
    pub data_dir: PathBuf,
    pub catalog: Catalog,
    pub tables: HashMap<TableId, DiskTable>,
}

impl StorageEngine {
    /// Prepare the data directory and load the last checkpoint (if any).
    /// Tables are attached afterwards by recovery via `attach_table`.
    pub fn open_dirs(data_dir: impl AsRef<Path>) -> Result<(Self, Option<CheckpointMeta>)> {
        let data_dir = data_dir.as_ref().to_path_buf();
        fs::create_dir_all(data_dir.join("tables"))?;
        fs::create_dir_all(data_dir.join("wal"))?;
        let tmp_dir = data_dir.join("tmp");
        if tmp_dir.exists() {
            let _ = fs::remove_dir_all(&tmp_dir);
        }
        let ckp_opt = CheckpointMeta::load_if_exists(&data_dir)?;
        Ok((
            Self {
                data_dir,
                catalog: Catalog::new(),
                tables: HashMap::new(),
            },
            ckp_opt,
        ))
    }

    pub fn mem_bytes(&self) -> usize {
        self.tables.values().map(|t| t.mem_bytes()).sum()
    }

    /// Open a table file during recovery and register it (catalog included).
    pub fn attach_table(
        &mut self,
        schema: TableSchema,
        generation: u64,
        next_row_id: RowId,
        safe_len: u64,
        committed: &HashSet<TxnId>,
    ) -> Result<()> {
        let left = mem_budget_bytes().saturating_sub(self.mem_bytes());
        self.catalog.insert_table(schema.clone())?;
        let dt = DiskTable::open_or_create(
            &self.data_dir,
            schema,
            generation,
            next_row_id,
            safe_len,
            committed,
            left,
        )?;
        self.tables.insert(dt.schema.table_id, dt);
        Ok(())
    }

    /// Create a brand-new, empty table.
    pub fn create_table(&mut self, schema: TableSchema) -> Result<()> {
        let tid = schema.table_id;
        self.catalog.insert_table(schema.clone())?;
        // A leftover file from an aborted earlier attempt must not leak rows into the new table.
        let stale = DiskTable::table_file_path(&self.data_dir, tid, 1);
        if stale.exists() {
            let _ = fs::remove_file(&stale);
        }
        let none = HashSet::new();
        let dt = DiskTable::open_or_create(&self.data_dir, schema, 1, 1, 0, &none, usize::MAX)?;
        self.tables.insert(tid, dt);
        Ok(())
    }

    /// Remove a table created by a transaction that failed before its commit point.
    pub fn drop_uncommitted_table(&mut self, table_id: TableId) {
        if let Some(dt) = self.tables.remove(&table_id) {
            let _ = fs::remove_file(&dt.file.path);
        }
        self.catalog.remove_table(table_id);
    }

    /// Keep the in-memory images within budget by moving the largest table to disk mode.
    pub fn enforce_memory_budget(&mut self) -> Result<()> {
        while self.mem_bytes() > mem_budget_bytes() {
            let Some(tid) = self
                .tables
                .iter()
                .filter(|(_, t)| t.is_mem())
                .max_by_key(|(_, t)| t.mem_bytes())
                .map(|(tid, _)| *tid)
            else {
                break;
            };
            self.get_table_mut(tid)?.spill_to_disk()?;
        }
        Ok(())
    }

    pub fn get_table(&self, table_id: TableId) -> Result<&DiskTable> {
        self.tables.get(&table_id).ok_or_else(|| {
            DbError::new(
                ErrorCode::TableNotFound,
                format!("table id {} missing in storage", table_id),
            )
        })
    }

    pub fn get_table_mut(&mut self, table_id: TableId) -> Result<&mut DiskTable> {
        self.tables.get_mut(&table_id).ok_or_else(|| {
            DbError::new(
                ErrorCode::TableNotFound,
                format!("table id {} missing in storage", table_id),
            )
        })
    }

    pub fn save_checkpoint_snapshot(
        &self,
        wal_seq: u64,
        wal_offset: u64,
        next_txn_id: u64,
    ) -> Result<()> {
        let mut snaps = Vec::with_capacity(self.tables.len());
        for dt in self.tables.values() {
            dt.file.sync_all()?;
            snaps.push(TableMetaSnapshot {
                schema: dt.schema.clone(),
                generation: dt.generation,
                safe_data_len: dt.file.len(),
                next_row_id: dt.next_row_id,
            });
        }
        let meta = CheckpointMeta {
            wal_seq,
            wal_offset,
            next_txn_id,
            next_table_id: self.catalog.next_table_id(),
            tables: snaps,
        };
        meta.save_atomic(&self.data_dir)
    }
}
