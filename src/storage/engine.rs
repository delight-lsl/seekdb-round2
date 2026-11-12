use crate::catalog::manager::Catalog;
use crate::catalog::schema::TableSchema;
use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::{Row, RowId, TableId, Value};
use crate::storage::codec::{
    decode_row, decode_row_selective, decode_schema, encode_row, encode_schema,
    extract_pk_from_row_bytes, extract_vector_from_row_bytes, write_u32, write_u64, Cursor,
};
use crate::storage::file::{atomic_write_file, sync_dir, TableDataFile};
use crate::storage::record::{encode_data_frame, DataRecordOp};
use crate::transaction::operation::WalOp;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Bounded FIFO/clock row cache capping heap RAM at `max_bytes`.
#[derive(Debug)]
pub struct BoundedRowCache {
    entries: HashMap<RowId, (Row, u32, u32)>,
    order: VecDeque<(RowId, u32)>,
    current_bytes: usize,
    max_bytes: usize,
    next_gen: u32,
}

impl BoundedRowCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            current_bytes: 0,
            max_bytes,
            next_gen: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (RowId, &Row)> {
        self.entries.iter().map(|(&rid, (row, _, _))| (rid, row))
    }

    pub fn keys(&self) -> impl Iterator<Item = RowId> + '_ {
        self.entries.keys().copied()
    }

    #[inline]
    pub fn get(&self, row_id: RowId) -> Option<&Row> {
        self.entries.get(&row_id).map(|(r, _, _)| r)
    }

    pub fn insert(&mut self, row_id: RowId, row: Row) {
        let est = estimate_row_bytes(&row);
        if est > self.max_bytes / 4 {
            return;
        }
        self.next_gen = self.next_gen.wrapping_add(1);
        let gen = self.next_gen;
        let est_u32 = est.min(u32::MAX as usize) as u32;
        if let Some((_, old_sz, _)) = self.entries.insert(row_id, (row, est_u32, gen)) {
            self.current_bytes = self.current_bytes.saturating_sub(old_sz as usize);
        }
        self.current_bytes += est;
        self.order.push_back((row_id, gen));
        while self.current_bytes > self.max_bytes {
            let Some((evict_id, evict_gen)) = self.order.pop_front() else {
                break;
            };
            if self
                .entries
                .get(&evict_id)
                .is_some_and(|&(_, _, g)| g == evict_gen)
            {
                if let Some((_, sz, _)) = self.entries.remove(&evict_id) {
                    self.current_bytes = self.current_bytes.saturating_sub(sz as usize);
                }
            }
        }
    }

    pub fn invalidate(&mut self, row_id: RowId) {
        if let Some((_, sz, _)) = self.entries.remove(&row_id) {
            self.current_bytes = self.current_bytes.saturating_sub(sz as usize);
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.current_bytes = 0;
    }

    pub fn shrink_to_fit(&mut self) {
        self.entries.shrink_to_fit();
        self.order.shrink_to_fit();
    }
}

fn estimate_row_bytes(row: &Row) -> usize {
    let mut bytes = 32;
    for v in &row.values {
        bytes += match v {
            Value::Null => 8,
            Value::BigInt(_) => 16,
            Value::Varchar(s) => 24 + s.len(),
            Value::Vector(vec) => 24 + vec.len() * 4,
        };
    }
    bytes
}

/// Disk-backed storage for a single table.
/// Only lightweight indexes (`pk_index`, `row_locations`) and a bounded row cache live on heap.
#[derive(Debug)]
pub struct DiskTable {
    pub schema: TableSchema,
    pub generation: u64,
    pub next_row_id: RowId,
    pub file: TableDataFile,
    pub row_locations: HashMap<RowId, (u64, u32)>,
    pub pk_index: HashMap<i64, RowId>,
    pub total_frames: u64,
    pub cache: BoundedRowCache,
}

impl DiskTable {
    pub fn table_file_path(data_dir: &Path, table_id: TableId, generation: u64) -> PathBuf {
        data_dir
            .join("tables")
            .join(format!("t_{}_g{}.dat", table_id, generation))
    }

    pub fn open_or_create(
        data_dir: &Path,
        schema: TableSchema,
        generation: u64,
        next_row_id: RowId,
        safe_data_len: Option<u64>,
    ) -> Result<Self> {
        let path = Self::table_file_path(data_dir, schema.table_id, generation);
        let mut file = TableDataFile::open(path)?;
        if let Some(safe_len) = safe_data_len {
            file.truncate(safe_len)?;
        }

        let mut row_locations = HashMap::new();
        let mut pk_index = HashMap::new();
        let mut max_row_id = 0_u64;
        let mut total_frames = 0_u64;
        let mut old_pks: HashMap<RowId, i64> = HashMap::new();

        let scan_len = file.len();
        let cache_in_single_pass = scan_len <= 32 * 1024 * 1024;
        let mut cache = BoundedRowCache::new(64 * 1024 * 1024);
        file.scan_frames(scan_len, |frame| {
            total_frames += 1;
            max_row_id = max_row_id.max(frame.row_id);
            match frame.op {
                DataRecordOp::Put => {
                    if let Some(old_pk) = old_pks.remove(&frame.row_id) {
                        pk_index.remove(&old_pk);
                    }
                    if cache_in_single_pass {
                        let row = decode_row(&frame.payload, &schema)?;
                        if let Some(pk) = pk_of_row(&schema, &row) {
                            pk_index.insert(pk, frame.row_id);
                            old_pks.insert(frame.row_id, pk);
                        }
                        cache.insert(frame.row_id, row);
                    } else if let Some(pk_idx) = schema.pk_col_idx {
                        if let Some(pk) = extract_pk_from_row_bytes(&frame.payload, pk_idx)? {
                            pk_index.insert(pk, frame.row_id);
                            old_pks.insert(frame.row_id, pk);
                        }
                    }
                    row_locations.insert(frame.row_id, (frame.frame_offset, frame.frame_len));
                }
                DataRecordOp::Tombstone => {
                    row_locations.remove(&frame.row_id);
                    if let Some(old_pk) = old_pks.remove(&frame.row_id) {
                        pk_index.remove(&old_pk);
                    }
                    if cache_in_single_pass {
                        cache.invalidate(frame.row_id);
                    }
                }
            }
            Ok(true)
        })?;

        Ok(Self {
            schema,
            generation,
            next_row_id: next_row_id.max(max_row_id + 1),
            file,
            row_locations,
            pk_index,
            total_frames,
            cache,
        })
    }

    pub fn live_count(&self) -> usize {
        self.row_locations.len()
    }

    #[inline]
    pub fn cache_covers_all(&self) -> bool {
        self.cache.len() == self.row_locations.len()
    }

    pub fn read_row(&mut self, row_id: RowId) -> Result<Row> {
        if let Some(cached) = self.cache.get(row_id) {
            return Ok(cached.clone());
        }
        let row = self.read_row_uncached(row_id)?;
        self.cache.insert(row_id, row.clone());
        Ok(row)
    }

    pub fn read_row_uncached(&self, row_id: RowId) -> Result<Row> {
        if let Some(cached) = self.cache.get(row_id) {
            return Ok(cached.clone());
        }
        let &(off, len) = self.row_locations.get(&row_id).ok_or_else(|| {
            DbError::new(
                ErrorCode::InternalError,
                format!("row_id {} not found", row_id),
            )
        })?;
        let frame = self.file.read_frame_at(off, len)?;
        decode_row(&frame.payload, &self.schema)
    }

    /// Read a single vector column from disk via `pread` (or cache) without decoding strings.
    pub fn read_vector_col(&self, row_id: RowId, col_idx: usize) -> Result<Option<Vec<f32>>> {
        if let Some(cached) = self.cache.get(row_id) {
            return match cached.values.get(col_idx) {
                Some(Value::Null) => Ok(None),
                Some(Value::Vector(v)) => Ok(Some(v.clone())),
                _ => Err(DbError::new(ErrorCode::InternalError, "invalid vector column")),
            };
        }
        let &(off, len) = self.row_locations.get(&row_id).ok_or_else(|| {
            DbError::new(
                ErrorCode::InternalError,
                format!("row_id {} not found", row_id),
            )
        })?;
        let frame = self.file.read_frame_at(off, len)?;
        extract_vector_from_row_bytes(&frame.payload, col_idx)
    }

    /// Read primary key from cache or `pread`.
    pub fn read_pk(&self, row_id: RowId) -> Result<Option<i64>> {
        let Some(pk_idx) = self.schema.pk_col_idx else {
            return Ok(None);
        };
        if let Some(cached) = self.cache.get(row_id) {
            return Ok(match cached.values.get(pk_idx) {
                Some(Value::BigInt(v)) => Some(*v),
                _ => None,
            });
        }
        let &(off, len) = self.row_locations.get(&row_id).ok_or_else(|| {
            DbError::new(
                ErrorCode::InternalError,
                format!("row_id {} not found", row_id),
            )
        })?;
        let frame = self.file.read_frame_at(off, len)?;
        extract_pk_from_row_bytes(&frame.payload, pk_idx)
    }

    /// Append a batch of inserts/updates/deletes cleanly to the data file.
    pub fn apply_batch(
        &mut self,
        puts: &[(RowId, Option<i64>, Row)],
        deletes: &[(RowId, Option<i64>)],
        sync_file: bool,
    ) -> Result<()> {
        if puts.is_empty() && deletes.is_empty() {
            return Ok(());
        }
        let mut batch_buf = Vec::new();
        let mut new_locs = Vec::with_capacity(puts.len());
        let mut payload_buf = Vec::new();
        let base_offset = self.file.len();

        for (rid, _, row) in puts {
            payload_buf.clear();
            encode_row(row, &mut payload_buf)?;
            let frame_off = base_offset + batch_buf.len() as u64;
            let frame_len =
                encode_data_frame(DataRecordOp::Put, *rid, &payload_buf, &mut batch_buf)?;
            new_locs.push((*rid, frame_off, frame_len, row));
        }
        for (rid, _) in deletes {
            encode_data_frame(DataRecordOp::Tombstone, *rid, &[], &mut batch_buf)?;
        }

        self.file.append_bytes(&batch_buf)?;
        if sync_file {
            self.file.sync()?;
        }

        for (_, old_pk, _) in puts {
            if let Some(opk) = old_pk {
                self.pk_index.remove(opk);
            }
        }
        for (rid, old_pk) in deletes {
            if let Some(opk) = old_pk {
                self.pk_index.remove(opk);
            }
            self.row_locations.remove(rid);
            self.total_frames += 1;
            self.cache.invalidate(*rid);
        }
        for (rid, off, len, row) in new_locs {
            self.row_locations.insert(rid, (off, len));
            self.total_frames += 1;
            self.next_row_id = self.next_row_id.max(rid + 1);
            if let Some(pk) = pk_of_row(&self.schema, row) {
                self.pk_index.insert(pk, rid);
            }
            self.cache.insert(rid, row.clone());
        }
        Ok(())
    }

    /// Append a batch reusing pre-encoded `WalOp::PutRow` bytes and moving `Row` into cache without re-encoding or cloning.
    pub fn apply_batch_owned_with_wal(
        &mut self,
        puts: Vec<(RowId, Option<i64>, Row)>,
        deletes: &[(RowId, Option<i64>)],
        wal_ops: &[WalOp],
        sync_file: bool,
    ) -> Result<()> {
        if puts.is_empty() && deletes.is_empty() {
            return Ok(());
        }
        let mut batch_buf = Vec::with_capacity(puts.len() * 64 + deletes.len() * 24);
        let mut new_locs = Vec::with_capacity(puts.len());
        let base_offset = self.file.len();

        let mut wal_put_iter = wal_ops.iter().filter_map(|op| match op {
            WalOp::PutRow {
                table_id,
                row_id,
                row_bytes,
                ..
            } if *table_id == self.schema.table_id => Some((*row_id, row_bytes.as_slice())),
            _ => None,
        });

        let mut fallback_buf = Vec::new();
        for (rid, old_pk, row) in puts {
            let payload_slice = match wal_put_iter.next() {
                Some((wrid, bytes)) if wrid == rid => bytes,
                _ => {
                    fallback_buf.clear();
                    encode_row(&row, &mut fallback_buf)?;
                    fallback_buf.as_slice()
                }
            };
            let frame_off = base_offset + batch_buf.len() as u64;
            let frame_len =
                encode_data_frame(DataRecordOp::Put, rid, payload_slice, &mut batch_buf)?;
            new_locs.push((rid, old_pk, frame_off, frame_len, row));
        }
        for (rid, _) in deletes {
            encode_data_frame(DataRecordOp::Tombstone, *rid, &[], &mut batch_buf)?;
        }

        self.file.append_bytes(&batch_buf)?;
        if sync_file {
            self.file.sync()?;
        }

        for (_, old_pk, _, _, _) in &new_locs {
            if let Some(opk) = old_pk {
                self.pk_index.remove(opk);
            }
        }
        for (rid, old_pk) in deletes {
            if let Some(opk) = old_pk {
                self.pk_index.remove(opk);
            }
            self.row_locations.remove(rid);
            self.total_frames += 1;
            self.cache.invalidate(*rid);
        }
        for (rid, _, off, len, row) in new_locs {
            self.row_locations.insert(rid, (off, len));
            self.total_frames += 1;
            self.next_row_id = self.next_row_id.max(rid + 1);
            if let Some(pk) = pk_of_row(&self.schema, &row) {
                self.pk_index.insert(pk, rid);
            }
            self.cache.insert(rid, row);
        }
        Ok(())
    }

    /// Sequential streaming scan over live rows, decoding `where_cols` first to skip
    /// unnecessary allocations on rows rejected by `WHERE`.
    pub fn scan_live_streaming<F>(
        &self,
        where_cols: &[bool],
        mut on_row: F,
    ) -> Result<()>
    where
        F: FnMut(RowId, &[u8], Row) -> Result<bool>,
    {
        let scan_len = self.file.len();
        self.file.scan_frames(scan_len, |frame| {
            if frame.op != DataRecordOp::Put {
                return Ok(true);
            }
            if self.row_locations.get(&frame.row_id) != Some(&(frame.frame_offset, frame.frame_len))
            {
                return Ok(true);
            }
            let partial = decode_row_selective(&frame.payload, &self.schema, where_cols)?;
            on_row(frame.row_id, &frame.payload, partial)
        })
    }

    /// Needs compaction if dead frames are >= 50% and file > 8 MiB.
    pub fn needs_compaction(&self) -> bool {
        let live = self.row_locations.len() as u64;
        self.file.len() > 8 * 1024 * 1024 && self.total_frames > live.saturating_mul(2) + 1024
    }

    /// Compact dead/superseded frames into a new generation file `t_<tid>_g<gen+1>.dat`.
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

        let mut new_locations = HashMap::with_capacity(self.row_locations.len());
        let mut total_live = 0_u64;
        {
            let out_file = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)?;
            let mut writer = std::io::BufWriter::with_capacity(256 * 1024, out_file);
            let mut cur_offset = 0_u64;
            let mut frame_buf = Vec::new();
            let scan_len = self.file.len();

            self.file.scan_frames(scan_len, |frame| {
                if frame.op == DataRecordOp::Put
                    && self.row_locations.get(&frame.row_id)
                        == Some(&(frame.frame_offset, frame.frame_len))
                {
                    frame_buf.clear();
                    let flen = encode_data_frame(
                        DataRecordOp::Put,
                        frame.row_id,
                        &frame.payload,
                        &mut frame_buf,
                    )?;
                    writer.write_all(&frame_buf)?;
                    new_locations.insert(frame.row_id, (cur_offset, flen));
                    cur_offset += flen as u64;
                    total_live += 1;
                }
                Ok(true)
            })?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }

        fs::rename(&tmp_path, &new_path)?;
        if let Some(p) = new_path.parent() {
            sync_dir(p)?;
        }

        self.file = TableDataFile::open(new_path)?;
        self.generation = next_gen;
        self.row_locations = new_locations;
        self.total_frames = total_live;
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

const CKP_MAGIC: [u8; 4] = *b"CKP1";
const CKP_VERSION: u32 = 1;

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
            return Err(DbError::new(
                ErrorCode::IoError,
                "checkpoint CRC mismatch",
            ));
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

/// Manages disk-backed tables and catalog metadata.
#[derive(Debug)]
pub struct StorageEngine {
    pub data_dir: PathBuf,
    pub catalog: Catalog,
    pub tables: HashMap<TableId, DiskTable>,
}

impl StorageEngine {
    pub fn open_from_checkpoint(data_dir: impl AsRef<Path>) -> Result<(Self, Option<CheckpointMeta>)> {
        let data_dir = data_dir.as_ref().to_path_buf();
        fs::create_dir_all(data_dir.join("tables"))?;
        fs::create_dir_all(data_dir.join("wal"))?;
        let tmp_dir = data_dir.join("tmp");
        if tmp_dir.exists() {
            let _ = fs::remove_dir_all(&tmp_dir);
        }
        fs::create_dir_all(&tmp_dir)?;

        let ckp_opt = CheckpointMeta::load_if_exists(&data_dir)?;
        let mut catalog = Catalog::new();
        let mut tables = HashMap::new();

        if let Some(ref ckp) = ckp_opt {
            let mut expected_files = std::collections::HashSet::new();
            let mut schemas = Vec::with_capacity(ckp.tables.len());
            for snap in &ckp.tables {
                schemas.push(snap.schema.clone());
                let dt = DiskTable::open_or_create(
                    &data_dir,
                    snap.schema.clone(),
                    snap.generation,
                    snap.next_row_id,
                    Some(snap.safe_data_len),
                )?;
                if let Some(fname) = dt.file.path.file_name().and_then(|s| s.to_str()) {
                    expected_files.insert(fname.to_string());
                }
                tables.insert(dt.schema.table_id, dt);
            }
            catalog = Catalog::from_state(schemas, ckp.next_table_id);

            // Clean up orphan .dat files created after the last checkpoint before a crash
            let tables_dir = data_dir.join("tables");
            if let Ok(entries) = fs::read_dir(&tables_dir) {
                for entry in entries.flatten() {
                    if let Some(name) = entry.file_name().to_str() {
                        if name.ends_with(".dat") && !expected_files.contains(name) {
                            let _ = fs::remove_file(entry.path());
                        }
                    }
                }
            }
        } else {
            // No checkpoint yet: remove any orphan .dat files from crashed uncommitted DDL
            let tables_dir = data_dir.join("tables");
            if let Ok(entries) = fs::read_dir(&tables_dir) {
                for entry in entries.flatten() {
                    if let Some(name) = entry.file_name().to_str() {
                        if name.ends_with(".dat") {
                            let _ = fs::remove_file(entry.path());
                        }
                    }
                }
            }
        }

        Ok((
            Self {
                data_dir,
                catalog,
                tables,
            },
            ckp_opt,
        ))
    }

    pub fn create_table(&mut self, schema: TableSchema) -> Result<()> {
        let tid = schema.table_id;
        self.catalog.insert_table(schema.clone())?;
        let dt = DiskTable::open_or_create(&self.data_dir, schema, 1, 1, None)?;
        self.tables.insert(tid, dt);
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
