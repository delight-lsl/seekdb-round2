//! Columnar in-memory image of a table.
//!
//! Every committed row of a table that fits the memory budget lives here exactly once:
//! vectors in fixed-size contiguous chunks (no reallocation, no per-row allocation),
//! strings in an append-only arena with a per-row hash for fast equality filters,
//! integers in plain arrays. Queries are answered from this image; the data file is
//! only written (and read once at startup).

use crate::catalog::schema::TableSchema;
use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::{Cell, DataType, Row, RowAccess, RowId, Value};
use crate::storage::codec::{for_each_raw_value, RawVal};
use crate::vector::distance::{l2_distance_squared_unchecked, scan_le, scan_topk_f32, sq_lower_bounds};
use std::collections::BinaryHeap;

const STR_CHUNK: usize = 256 * 1024;
const VEC_CHUNK_BYTES: usize = 1024 * 1024;
const NULL_LEN: u32 = u32::MAX;
/// Tables smaller than this are scanned directly; larger ones get quantized codes.
const SQ_MIN_ROWS: usize = 1024;

/// 7-bit scalar-quantized copy of a vector column used as a *lower-bound* filter.
///
/// Each component is mapped to the cell `floor((x - lo) / step)` clamped to `0..=127`.
/// For two values in cells `ca < cb` the true difference is at least `(cb - ca - 1) * step`
/// (cells are intervals; clamping only makes the end cells larger), so
/// `sum(max(0, |ca - cb| - 1)^2) * step^2` never exceeds the true squared distance.
/// The scan discards a row only when this bound already exceeds the current k-th best
/// exact distance, and evaluates every remaining row with the exact kernel, so results
/// are identical to a full exact scan.
#[derive(Debug)]
pub struct SqCodes {
    lo: f32,
    inv_step: f32,
    /// Squared cell width, shrunk to absorb floating-point error in the cell boundaries.
    safe_step_sq: f64,
    /// Bytes per row (dimension rounded up to a multiple of 32, zero padded).
    stride: usize,
    chunks: Vec<Vec<u8>>,
    /// Components that fell outside the quantized range (clamped; still valid, less tight).
    out_of_range: usize,
    encoded: usize,
}

impl SqCodes {
    #[inline]
    fn encode(&mut self, v: &[f32], chunk: usize, row: usize) {
        let lo = self.lo;
        let inv = self.inv_step;
        let dst = &mut self.chunks[chunk][row * self.stride..row * self.stride + v.len()];
        let mut oor = 0usize;
        for (o, &x) in dst.iter_mut().zip(v.iter()) {
            let t = (x - lo) * inv;
            oor += (t < 0.0 || t >= 128.0) as usize;
            *o = t.max(0.0).min(127.0) as u8;
        }
        self.out_of_range += oor;
        self.encoded += v.len();
    }

    fn encode_query(&self, q: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; self.stride];
        for (o, &x) in out.iter_mut().zip(q.iter()) {
            let t = (x - self.lo) * self.inv_step;
            *o = t.max(0.0).min(127.0) as u8;
        }
        out
    }

    fn is_stale(&self) -> bool {
        self.out_of_range > self.encoded / 64 + 256
    }
}

#[derive(Debug, Clone, Copy)]
struct StrRef {
    chunk: u32,
    off: u32,
    len: u32,
}

const NULL_REF: StrRef = StrRef {
    chunk: 0,
    off: 0,
    len: NULL_LEN,
};

#[inline]
pub fn hash_bytes(b: &[u8]) -> u64 {
    // FNV-1a with a final avalanche; only used as an equality pre-filter.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &c in b {
        h ^= c as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^ (h >> 29)
}

#[derive(Debug)]
pub struct StrCol {
    refs: Vec<StrRef>,
    hashes: Vec<u64>,
    chunks: Vec<Vec<u8>>,
    live_bytes: usize,
    used_bytes: usize,
}

impl StrCol {
    fn new() -> Self {
        Self {
            refs: Vec::new(),
            hashes: Vec::new(),
            chunks: Vec::new(),
            live_bytes: 0,
            used_bytes: 0,
        }
    }

    #[inline]
    fn get(&self, slot: usize) -> Option<&[u8]> {
        let r = self.refs[slot];
        if r.len == NULL_LEN {
            return None;
        }
        let c = &self.chunks[r.chunk as usize];
        Some(&c[r.off as usize..r.off as usize + r.len as usize])
    }

    fn store(&mut self, bytes: &[u8]) -> StrRef {
        let need_new = match self.chunks.last() {
            Some(c) => c.capacity() - c.len() < bytes.len(),
            None => true,
        };
        if need_new {
            self.chunks.push(Vec::with_capacity(STR_CHUNK.max(bytes.len())));
        }
        let ci = self.chunks.len() - 1;
        let c = &mut self.chunks[ci];
        let off = c.len();
        c.extend_from_slice(bytes);
        self.live_bytes += bytes.len();
        self.used_bytes += bytes.len();
        StrRef {
            chunk: ci as u32,
            off: off as u32,
            len: bytes.len() as u32,
        }
    }

    fn set(&mut self, slot: usize, val: Option<&[u8]>) {
        let old = self.refs[slot];
        if old.len != NULL_LEN {
            self.live_bytes -= old.len as usize;
        }
        match val {
            Some(b) => {
                let r = self.store(b);
                self.refs[slot] = r;
                self.hashes[slot] = hash_bytes(b);
            }
            None => {
                self.refs[slot] = NULL_REF;
                self.hashes[slot] = 0;
            }
        }
    }

    /// Rewrite the arena when more than half of it is garbage left by updates/deletes.
    fn maybe_compact(&mut self) {
        if self.used_bytes <= self.live_bytes * 2 + STR_CHUNK * 4 {
            return;
        }
        let old_chunks = std::mem::take(&mut self.chunks);
        self.live_bytes = 0;
        self.used_bytes = 0;
        for slot in 0..self.refs.len() {
            let r = self.refs[slot];
            if r.len == NULL_LEN {
                continue;
            }
            let src = &old_chunks[r.chunk as usize][r.off as usize..r.off as usize + r.len as usize];
            let nr = self.store(src);
            self.refs[slot] = nr;
        }
    }
}

#[derive(Debug)]
pub struct VecCol {
    pub dim: usize,
    rows_per_chunk: usize,
    chunks: Vec<Vec<f32>>,
    nulls: Vec<bool>,
    pub null_count: usize,
    /// Largest absolute component ever stored (monotone upper bound).
    pub max_abs: f32,
    sq: Option<SqCodes>,
    /// Do not retry building codes before the column has this many slots.
    sq_retry_at: usize,
}

impl VecCol {
    fn new(dim: usize) -> Self {
        let dim = dim.max(1);
        Self {
            dim,
            rows_per_chunk: (VEC_CHUNK_BYTES / (dim * 4)).max(1),
            chunks: Vec::new(),
            nulls: Vec::new(),
            null_count: 0,
            max_abs: 0.0,
            sq: None,
            sq_retry_at: 0,
        }
    }

    fn push_slot(&mut self) {
        let slot = self.nulls.len();
        if slot % self.rows_per_chunk == 0 {
            self.chunks
                .push(Vec::with_capacity(self.rows_per_chunk * self.dim));
        }
        if let Some(sq) = &mut self.sq {
            if slot % self.rows_per_chunk == 0 {
                sq.chunks
                    .push(Vec::with_capacity(self.rows_per_chunk * sq.stride));
            }
            let c = sq.chunks.last_mut().expect("code chunk exists");
            let new_len = c.len() + sq.stride;
            c.resize(new_len, 0);
        }
        let c = self.chunks.last_mut().expect("chunk exists");
        let new_len = c.len() + self.dim;
        c.resize(new_len, 0.0);
        self.nulls.push(true);
        self.null_count += 1;
    }

    #[inline]
    pub fn get(&self, slot: usize) -> &[f32] {
        let c = &self.chunks[slot / self.rows_per_chunk];
        let s = (slot % self.rows_per_chunk) * self.dim;
        &c[s..s + self.dim]
    }

    #[inline]
    fn get_mut(&mut self, slot: usize) -> &mut [f32] {
        let dim = self.dim;
        let c = &mut self.chunks[slot / self.rows_per_chunk];
        let s = (slot % self.rows_per_chunk) * dim;
        &mut c[s..s + dim]
    }

    #[inline]
    pub fn is_null(&self, slot: usize) -> bool {
        self.nulls[slot]
    }

    fn mark(&mut self, slot: usize, null: bool) {
        if self.nulls[slot] != null {
            self.nulls[slot] = null;
            if null {
                self.null_count += 1;
            } else {
                self.null_count -= 1;
            }
        }
    }

    /// Bookkeeping after the vector of `slot` was (re)written.
    fn note_max(&mut self, slot: usize) {
        let rpc = self.rows_per_chunk;
        let dim = self.dim;
        let c = &self.chunks[slot / rpc];
        let v = &c[(slot % rpc) * dim..(slot % rpc) * dim + dim];
        let mut m = self.max_abs;
        for &x in v {
            let a = x.abs();
            if a > m {
                m = a;
            }
        }
        self.max_abs = m;
        if let Some(sq) = &mut self.sq {
            sq.encode(v, slot / rpc, slot % rpc);
        }
    }

    fn code_bytes(&self) -> usize {
        self.sq.as_ref().map_or(0, |s| s.stride * self.nulls.len())
    }

    /// Build (or rebuild) the quantized codes when the column is large enough and
    /// its value range is suitable. Cheap no-op otherwise.
    fn prepare_sq(&mut self, live: usize) {
        let nslots = self.nulls.len();
        match &self.sq {
            Some(sq) if !sq.is_stale() => return,
            Some(_) => {}
            None => {
                if live < SQ_MIN_ROWS || nslots < self.sq_retry_at {
                    return;
                }
            }
        }
        self.sq = None;
        self.sq_retry_at = nslots * 2 + SQ_MIN_ROWS;
        if self.dim > 32768 {
            return;
        }
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for slot in 0..nslots {
            if self.nulls[slot] {
                continue;
            }
            for &x in self.get(slot) {
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        let range = (hi as f64 - lo as f64) * 1.04;
        let lo_w = lo as f64 - 0.02 * (hi as f64 - lo as f64);
        let mag = lo.abs().max(hi.abs()) as f64;
        // The cell width must dwarf the rounding error of `(x - lo) * inv_step`.
        if !(range.is_finite() && range > 0.0 && mag < 1e15 && range * 256.0 >= mag) {
            return;
        }
        let step = range / 128.0;
        // Cell boundaries computed in single precision are off by at most a few ulps of the
        // largest magnitude involved; shrinking the cell width by twice a generous bound on
        // that error keeps `(cells apart - 1) * width` a true lower bound.
        let err = 8.0 * f32::EPSILON as f64 * mag.max(range);
        let safe_step = step - 2.0 * err;
        let stride = (self.dim + 31) & !31;
        let mut sq = SqCodes {
            lo: lo_w as f32,
            inv_step: (1.0 / step) as f32,
            safe_step_sq: safe_step * safe_step * (1.0 - 1e-9),
            stride,
            chunks: Vec::with_capacity(self.chunks.len()),
            out_of_range: 0,
            encoded: 0,
        };
        let rpc = self.rows_per_chunk;
        for (ci, c) in self.chunks.iter().enumerate() {
            let rows = c.len() / self.dim;
            let mut codes = Vec::with_capacity(rpc * stride);
            codes.resize(rows * stride, 0u8);
            sq.chunks.push(codes);
            for r in 0..rows {
                if !self.nulls[ci * rpc + r] {
                    sq.encode(&c[r * self.dim..(r + 1) * self.dim], ci, r);
                }
            }
        }
        sq.out_of_range = 0;
        self.sq = Some(sq);
        self.sq_retry_at = 0;
    }

    /// Iterate `(first_slot, rows, contiguous_values)` blocks.
    pub fn blocks(&self) -> impl Iterator<Item = (usize, usize, &[f32])> {
        let rpc = self.rows_per_chunk;
        let dim = self.dim;
        self.chunks
            .iter()
            .enumerate()
            .map(move |(ci, c)| (ci * rpc, c.len() / dim, c.as_slice()))
    }
}

#[derive(Debug)]
pub enum MemCol {
    Int { vals: Vec<i64>, nulls: Vec<bool> },
    Str(StrCol),
    Vector(VecCol),
}

#[derive(Debug)]
pub struct MemTable {
    cols: Vec<MemCol>,
    /// slot -> row id (0 marks a free slot; row ids start at 1).
    rids: Vec<RowId>,
    free: Vec<u32>,
    live: usize,
    pk_col: Option<usize>,
    slot_fixed_bytes: usize,
}

impl MemTable {
    pub fn new(schema: &TableSchema) -> Self {
        // Row id plus the row-id -> slot and primary-key -> row-id map entries.
        let mut slot_fixed_bytes = 8 + 40;
        let cols = schema
            .columns
            .iter()
            .map(|c| match c.data_type {
                DataType::BigInt => {
                    slot_fixed_bytes += 9;
                    MemCol::Int {
                        vals: Vec::new(),
                        nulls: Vec::new(),
                    }
                }
                DataType::Varchar(_) => {
                    slot_fixed_bytes += 20;
                    MemCol::Str(StrCol::new())
                }
                DataType::Vector(d) => {
                    slot_fixed_bytes += d * 4 + 1;
                    MemCol::Vector(VecCol::new(d))
                }
            })
            .collect();
        Self {
            cols,
            rids: Vec::new(),
            free: Vec::new(),
            live: 0,
            pk_col: schema.pk_col_idx,
            slot_fixed_bytes,
        }
    }

    #[inline]
    pub fn nslots(&self) -> usize {
        self.rids.len()
    }

    #[inline]
    pub fn live(&self) -> usize {
        self.live
    }

    #[inline]
    pub fn has_holes(&self) -> bool {
        self.live != self.rids.len()
    }

    #[inline]
    pub fn rid(&self, slot: usize) -> RowId {
        self.rids[slot]
    }

    #[inline]
    pub fn rids(&self) -> &[RowId] {
        &self.rids
    }

    /// Approximate heap footprint.
    pub fn bytes(&self) -> usize {
        let mut b = self.rids.len() * self.slot_fixed_bytes;
        for c in &self.cols {
            match c {
                MemCol::Str(s) => b += s.used_bytes,
                MemCol::Vector(v) => b += v.code_bytes(),
                MemCol::Int { .. } => {}
            }
        }
        b
    }

    /// Make the vector column `col` ready for searching (maintains the quantized codes).
    pub fn prepare_vector_search(&mut self, col: usize) {
        let live = self.live;
        if let Some(MemCol::Vector(v)) = self.cols.get_mut(col) {
            v.prepare_sq(live);
        }
    }

    pub fn alloc_slot(&mut self) -> u32 {
        if let Some(s) = self.free.pop() {
            return s;
        }
        let slot = self.rids.len();
        self.rids.push(0);
        for c in &mut self.cols {
            match c {
                MemCol::Int { vals, nulls } => {
                    vals.push(0);
                    nulls.push(true);
                }
                MemCol::Str(s) => {
                    s.refs.push(NULL_REF);
                    s.hashes.push(0);
                }
                MemCol::Vector(v) => v.push_slot(),
            }
        }
        slot as u32
    }

    fn occupy(&mut self, slot: u32, rid: RowId) {
        let s = slot as usize;
        if self.rids[s] == 0 {
            self.live += 1;
        }
        self.rids[s] = rid;
    }

    pub fn free_slot(&mut self, slot: u32) {
        let s = slot as usize;
        if self.rids[s] == 0 {
            return;
        }
        self.rids[s] = 0;
        self.live -= 1;
        for c in &mut self.cols {
            match c {
                MemCol::Int { nulls, .. } => nulls[s] = true,
                MemCol::Str(sc) => sc.set(s, None),
                MemCol::Vector(v) => v.mark(s, true),
            }
        }
        self.free.push(slot);
    }

    pub fn set_row(&mut self, slot: u32, rid: RowId, row: &Row) -> Result<()> {
        let s = slot as usize;
        if row.values.len() != self.cols.len() {
            return Err(DbError::new(ErrorCode::InternalError, "row width mismatch"));
        }
        for (c, v) in self.cols.iter_mut().zip(row.values.iter()) {
            match (c, v) {
                (MemCol::Int { vals, nulls }, Value::BigInt(x)) => {
                    vals[s] = *x;
                    nulls[s] = false;
                }
                (MemCol::Int { nulls, .. }, Value::Null) => nulls[s] = true,
                (MemCol::Str(sc), Value::Varchar(x)) => sc.set(s, Some(x.as_bytes())),
                (MemCol::Str(sc), Value::Null) => sc.set(s, None),
                (MemCol::Vector(vc), Value::Vector(x)) => {
                    if x.len() != vc.dim {
                        return Err(DbError::new(ErrorCode::InternalError, "vector width mismatch"));
                    }
                    vc.get_mut(s).copy_from_slice(x);
                    vc.mark(s, false);
                    vc.note_max(s);
                }
                (MemCol::Vector(vc), Value::Null) => vc.mark(s, true),
                _ => {
                    return Err(DbError::new(ErrorCode::InternalError, "column type mismatch"));
                }
            }
        }
        self.occupy(slot, rid);
        Ok(())
    }

    /// Load a row straight from its encoded bytes (no intermediate `Row`).
    pub fn set_from_bytes(&mut self, slot: u32, rid: RowId, payload: &[u8]) -> Result<()> {
        let s = slot as usize;
        let cols = &mut self.cols;
        for_each_raw_value(payload, cols.len(), |idx, raw| {
            match (&mut cols[idx], raw) {
                (MemCol::Int { vals, nulls }, RawVal::Int(x)) => {
                    vals[s] = x;
                    nulls[s] = false;
                }
                (MemCol::Int { nulls, .. }, RawVal::Null) => nulls[s] = true,
                (MemCol::Str(sc), RawVal::Str(b)) => {
                    if std::str::from_utf8(b).is_err() {
                        return Err(DbError::new(ErrorCode::IoError, "invalid UTF-8 in stored string"));
                    }
                    sc.set(s, Some(b));
                }
                (MemCol::Str(sc), RawVal::Null) => sc.set(s, None),
                (MemCol::Vector(vc), RawVal::Vector(b)) => {
                    if b.len() != vc.dim * 4 {
                        return Err(DbError::new(ErrorCode::IoError, "stored vector dimension mismatch"));
                    }
                    for (d, chunk) in vc.get_mut(s).iter_mut().zip(b.chunks_exact(4)) {
                        *d = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                    vc.mark(s, false);
                    vc.note_max(s);
                }
                (MemCol::Vector(vc), RawVal::Null) => vc.mark(s, true),
                _ => {
                    return Err(DbError::new(ErrorCode::IoError, "stored column type mismatch"));
                }
            }
            Ok(())
        })?;
        self.occupy(slot, rid);
        Ok(())
    }

    pub fn maybe_compact(&mut self) {
        for c in &mut self.cols {
            if let MemCol::Str(s) = c {
                s.maybe_compact();
            }
        }
    }

    #[inline]
    pub fn pk(&self, slot: u32) -> Option<i64> {
        let idx = self.pk_col?;
        match &self.cols[idx] {
            MemCol::Int { vals, nulls } => {
                if nulls[slot as usize] {
                    None
                } else {
                    Some(vals[slot as usize])
                }
            }
            _ => None,
        }
    }

    #[inline]
    pub fn cell(&self, slot: u32, col: usize) -> Cell<'_> {
        let s = slot as usize;
        match &self.cols[col] {
            MemCol::Int { vals, nulls } => {
                if nulls[s] {
                    Cell::Null
                } else {
                    Cell::Int(vals[s])
                }
            }
            MemCol::Str(sc) => match sc.get(s) {
                // SAFETY: only valid UTF-8 is ever stored (checked on every load path).
                Some(b) => Cell::Str(unsafe { std::str::from_utf8_unchecked(b) }),
                None => Cell::Null,
            },
            MemCol::Vector(vc) => {
                if vc.is_null(s) {
                    Cell::Null
                } else {
                    Cell::Vector(vc.get(s))
                }
            }
        }
    }

    #[inline]
    pub fn row(&self, slot: u32) -> SlotRef<'_> {
        SlotRef { mem: self, slot }
    }

    pub fn vec_col(&self, col: usize) -> Option<&VecCol> {
        match &self.cols[col] {
            MemCol::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Materialize a `Row`; columns not in `needed` become NULL placeholders.
    pub fn materialize(&self, slot: u32, needed: Option<&[bool]>) -> Row {
        let mut values = Vec::with_capacity(self.cols.len());
        for idx in 0..self.cols.len() {
            if needed.is_some_and(|n| !n[idx]) {
                values.push(Value::Null);
                continue;
            }
            values.push(match self.cell(slot, idx) {
                Cell::Null => Value::Null,
                Cell::Int(v) => Value::BigInt(v),
                Cell::Str(s) => Value::Varchar(s.to_string()),
                Cell::Vector(v) => Value::Vector(v.to_vec()),
            });
        }
        Row { values }
    }

    /// `mask[slot] &= (column == literal)` for a plain equality predicate.
    /// Returns false when the column/literal combination is not handled here.
    pub fn and_eq_mask(&self, col: usize, lit: &Value, mask: &mut [u8]) -> bool {
        match (&self.cols[col], lit) {
            (MemCol::Int { vals, nulls }, Value::BigInt(x)) => {
                for ((m, v), n) in mask.iter_mut().zip(vals.iter()).zip(nulls.iter()) {
                    *m &= (*v == *x && !*n) as u8;
                }
                true
            }
            (MemCol::Str(sc), Value::Varchar(x)) => {
                let xb = x.as_bytes();
                let h = hash_bytes(xb);
                // Branch-free hash filter, then a byte comparison of the survivors.
                for (m, hh) in mask.iter_mut().zip(sc.hashes.iter()) {
                    *m &= (*hh == h) as u8;
                }
                for (ci, chunk) in mask.chunks_mut(8).enumerate() {
                    if chunk.iter().all(|&b| b == 0) {
                        continue;
                    }
                    for (j, m) in chunk.iter_mut().enumerate() {
                        if *m != 0 && sc.get(ci * 8 + j) != Some(xb) {
                            *m = 0;
                        }
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// Base selection mask: live rows whose vector column `col` is not NULL.
    pub fn base_mask(&self, vec_col: Option<usize>) -> Vec<u8> {
        let mut mask: Vec<u8> = if self.has_holes() {
            self.rids.iter().map(|&r| (r != 0) as u8).collect()
        } else {
            vec![1u8; self.rids.len()]
        };
        if let Some(c) = vec_col {
            if let MemCol::Vector(v) = &self.cols[c] {
                if v.null_count > 0 {
                    for (m, n) in mask.iter_mut().zip(v.nulls.iter()) {
                        *m &= !*n as u8;
                    }
                }
            }
        }
        mask
    }

    /// Exact top-k nearest rows by `(squared L2 distance in f64, primary key, row id)`.
    ///
    /// `mask` (when given) selects the candidate slots and must already exclude free
    /// slots and NULL vectors; `None` means "every slot" and is only valid when the
    /// table has no holes and the column has no NULLs.
    ///
    /// Large columns are scanned through the quantized lower-bound codes: a row is
    /// skipped only if its lower bound already exceeds the current k-th best exact
    /// distance; everything else is evaluated with the exact double-precision kernel.
    /// Smaller columns use a single-precision SIMD pass that selects a provable superset
    /// (rounding slack included) which is then ranked exactly. Either way the result is
    /// identical to a full double-precision scan.
    pub fn topk_exact(
        &self,
        col: usize,
        query: &[f32],
        k: usize,
        mask: Option<&[u8]>,
    ) -> Vec<(f64, Option<i64>, RowId, u32)> {
        let Some(vc) = self.vec_col(col) else {
            return Vec::new();
        };
        let nslots = self.rids.len();
        if k == 0 || query.len() != vc.dim || nslots == 0 {
            return Vec::new();
        }
        let q_max = query.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
        let mut out: Vec<(f64, Option<i64>, RowId, u32)>;

        if let (Some(sq), true) = (&vc.sq, k.saturating_mul(8) <= nslots && q_max < 1e15) {
            // Pass 0: integer lower bound of every candidate row.
            let qcodes = sq.encode_query(query);
            let mut lbs: Vec<u32> = Vec::with_capacity(if mask.is_some() { nslots / 4 + 64 } else { nslots });
            let mut sel: Vec<u32> = Vec::new();
            let rpc = vc.rows_per_chunk;
            for (ci, codes) in sq.chunks.iter().enumerate() {
                let first = ci * rpc;
                let rows = codes.len() / sq.stride;
                let m = mask.map(|m| &m[first..first + rows]);
                sq_lower_bounds(codes, sq.stride, rows, &qcodes, m, first as u32, &mut lbs, &mut sel);
            }
            let slot_at = |i: usize| -> u32 {
                if mask.is_some() {
                    sel[i]
                } else {
                    i as u32
                }
            };
            let exact = |slot: u32| -> (f64, Option<i64>, RowId, u32) {
                (
                    l2_distance_squared_unchecked(vc.get(slot as usize), query),
                    self.pk(slot),
                    self.rids[slot as usize],
                    slot,
                )
            };
            if lbs.len() <= k {
                out = (0..lbs.len()).map(|i| exact(slot_at(i))).collect();
            } else {
                // Pass 1: the k rows with the smallest lower bounds give a first exact
                // k-th distance (an upper bound of the final one).
                let mut best: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(k + 1);
                scan_le(&lbs, u32::MAX, |i| {
                    let item = (lbs[i], i as u32);
                    if best.len() < k {
                        best.push(item);
                    } else if item < *best.peek().unwrap() {
                        *best.peek_mut().unwrap() = item;
                    }
                    if best.len() < k {
                        u32::MAX
                    } else {
                        best.peek().unwrap().0
                    }
                });
                let mut kth = 0.0_f64;
                for &(_, i) in best.iter() {
                    kth = kth.max(exact(slot_at(i as usize)).0);
                }
                // Pass 2: only rows whose lower bound does not exceed that distance can
                // tie or beat it; evaluate exactly those.
                let t = (kth / sq.safe_step_sq).floor();
                let thr = if t >= i32::MAX as f64 { i32::MAX as u32 } else { t as u32 };
                out = Vec::with_capacity(k * 8);
                scan_le(&lbs, thr, |i| {
                    out.push(exact(slot_at(i)));
                    thr
                });
            }
        } else if vc.max_abs < 1e15 && q_max < 1e15 {
            let cands = scan_topk_f32(vc.blocks(), vc.dim, query, k, mask);
            out = Vec::with_capacity(cands.len());
            for slot in cands {
                let d = l2_distance_squared_unchecked(vc.get(slot as usize), query);
                out.push((d, self.pk(slot), self.rids[slot as usize], slot));
            }
        } else {
            out = Vec::new();
            for slot in 0..nslots {
                if let Some(m) = mask {
                    if m[slot] == 0 {
                        continue;
                    }
                }
                let d = l2_distance_squared_unchecked(vc.get(slot), query);
                out.push((d, self.pk(slot as u32), self.rids[slot], slot as u32));
                if out.len() >= k.saturating_mul(4).max(1024) {
                    sort_cands(&mut out);
                    out.truncate(k);
                }
            }
        }
        sort_cands(&mut out);
        out.truncate(k);
        out
    }
}

fn sort_cands(v: &mut [(f64, Option<i64>, RowId, u32)]) {
    v.sort_unstable_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
}

/// Borrowed view of one row inside a `MemTable`.
#[derive(Clone, Copy)]
pub struct SlotRef<'a> {
    mem: &'a MemTable,
    slot: u32,
}

impl RowAccess for SlotRef<'_> {
    #[inline]
    fn cell(&self, idx: usize) -> Cell<'_> {
        self.mem.cell(self.slot, idx)
    }
}
