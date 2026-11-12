use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::{RowId, Value};
use crate::storage::codec::Cursor;
use crate::storage::engine::DiskTable;
use crate::storage::file::sync_dir;
use crate::vector::distance::{l2_distance_squared_f32_fast, l2_distance_squared_unchecked};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

const HNSW_MAGIC: [u8; 4] = *b"HNS1";
const MAX_INMEM_VECTOR_BYTES: usize = 512 * 1024 * 1024; // 512 MiB heap budget for HNSW vectors
const COMPACT_NEIGHBORS_FLAG: u8 = 0x80;

struct CrcWriter<W: Write> {
    inner: BufWriter<W>,
    hasher: crc32fast::Hasher,
}

impl<W: Write> CrcWriter<W> {
    fn new(writer: W) -> Self {
        Self {
            inner: BufWriter::with_capacity(64 * 1024, writer),
            hasher: crc32fast::Hasher::new(),
        }
    }

    #[inline]
    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.hasher.update(bytes);
        self.inner.write_all(bytes)?;
        Ok(())
    }

    #[inline]
    fn write_u8(&mut self, v: u8) -> Result<()> {
        self.write_bytes(&[v])
    }

    #[inline]
    fn write_u16(&mut self, v: u16) -> Result<()> {
        self.write_bytes(&v.to_le_bytes())
    }

    #[inline]
    fn write_u32(&mut self, v: u32) -> Result<()> {
        self.write_bytes(&v.to_le_bytes())
    }

    #[inline]
    fn write_u64(&mut self, v: u64) -> Result<()> {
        self.write_bytes(&v.to_le_bytes())
    }

    #[inline]
    fn write_i64(&mut self, v: i64) -> Result<()> {
        self.write_bytes(&v.to_le_bytes())
    }

    fn finish(mut self) -> Result<W> {
        let crc = self.hasher.finalize();
        self.inner.write_all(&crc.to_le_bytes())?;
        self.inner.flush()?;
        self.inner
            .into_inner()
            .map_err(|e| DbError::from(e.into_error()))
    }
}

#[inline]
fn is_slice_safe_for_f32(v: &[f32]) -> bool {
    v.iter().all(|&x| x.abs() < 1e15)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DistNode {
    dist: f64,
    node_id: u32,
}

impl Eq for DistNode {}

impl PartialOrd for DistNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

/// Vector storage for HNSW supporting fast contiguous RAM mode up to 512 MiB
/// and automatic disk-backed LRU mode (`pread`) for datasets exceeding RAM limits.
#[derive(Debug)]
struct VectorStore {
    dim: usize,
    in_mem: Vec<f32>,
    spilled: bool,
    spill_path: PathBuf,
    spill_file: Option<File>,
    lru_map: HashMap<u32, (Vec<f32>, u64)>,
    lru_order: VecDeque<(u32, u64)>,
    lru_gen: u64,
    max_cached_nodes: usize,
    max_inmem_bytes: usize,
    safe_for_f32_dist: bool,
}

impl VectorStore {
    fn new(dim: usize, spill_path: PathBuf) -> Self {
        Self::with_limit(dim, spill_path, MAX_INMEM_VECTOR_BYTES)
    }

    fn with_limit(dim: usize, spill_path: PathBuf, max_inmem_bytes: usize) -> Self {
        let vec_bytes = dim.saturating_mul(4).max(4);
        let max_cached_nodes = (max_inmem_bytes / vec_bytes).max(64);
        Self {
            dim,
            in_mem: Vec::new(),
            spilled: false,
            spill_path,
            spill_file: None,
            lru_map: HashMap::new(),
            lru_order: VecDeque::new(),
            lru_gen: 0,
            max_cached_nodes,
            max_inmem_bytes,
            safe_for_f32_dist: true,
        }
    }

    fn append(&mut self, node_id: u32, vec: &[f32]) -> Result<()> {
        debug_assert_eq!(vec.len(), self.dim);
        if self.safe_for_f32_dist && !is_slice_safe_for_f32(vec) {
            self.safe_for_f32_dist = false;
        }
        if !self.spilled {
            let needed_bytes = (node_id as usize + 1) * self.dim * 4;
            if needed_bytes <= self.max_inmem_bytes {
                self.in_mem.extend_from_slice(vec);
                return Ok(());
            }
            self.transition_to_spill()?;
        }
        let file = self.spill_file.as_mut().expect("spill_file opened");
        let offset = (node_id as u64) * (self.dim as u64) * 4;
        let mut raw = Vec::with_capacity(self.dim * 4);
        for &x in vec {
            raw.extend_from_slice(&x.to_le_bytes());
        }
        file.write_all_at(&raw, offset)?;
        self.insert_lru(node_id, vec.to_vec());
        Ok(())
    }

    fn transition_to_spill(&mut self) -> Result<()> {
        if let Some(p) = self.spill_path.parent() {
            fs::create_dir_all(p)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&self.spill_path)?;
        let mut writer = BufWriter::with_capacity(256 * 1024, &file);
        for &x in &self.in_mem {
            writer.write_all(&x.to_le_bytes())?;
        }
        writer.flush()?;
        drop(writer);
        // Move up to max_cached_nodes recent vectors into LRU cache and free contiguous vec
        let total_nodes = self.in_mem.len() / self.dim;
        let keep_start = total_nodes.saturating_sub(self.max_cached_nodes);
        for nid in keep_start..total_nodes {
            let start = nid * self.dim;
            let v = self.in_mem[start..start + self.dim].to_vec();
            self.insert_lru(nid as u32, v);
        }
        self.in_mem = Vec::new();
        self.in_mem.shrink_to_fit();
        self.spill_file = Some(file);
        self.spilled = true;
        Ok(())
    }

    fn insert_lru(&mut self, node_id: u32, vec: Vec<f32>) {
        if self.lru_map.contains_key(&node_id) {
            self.lru_gen = self.lru_gen.wrapping_add(1);
            let g = self.lru_gen;
            self.lru_map.insert(node_id, (vec, g));
            self.lru_order.push_back((node_id, g));
            return;
        }
        while self.lru_map.len() >= self.max_cached_nodes {
            let Some((evict, egen)) = self.lru_order.pop_front() else {
                break;
            };
            if self
                .lru_map
                .get(&evict)
                .is_some_and(|&(_, g)| g == egen)
            {
                self.lru_map.remove(&evict);
            }
        }
        self.lru_gen = self.lru_gen.wrapping_add(1);
        let g = self.lru_gen;
        self.lru_order.push_back((node_id, g));
        self.lru_map.insert(node_id, (vec, g));
    }

    #[inline]
    fn prefetch_node(&self, node_id: u32) {
        #[cfg(target_arch = "x86_64")]
        {
            if !self.spilled {
                let start = (node_id as usize) * self.dim;
                if start < self.in_mem.len() {
                    unsafe {
                        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                        let ptr = self.in_mem.as_ptr().add(start) as *const i8;
                        _mm_prefetch(ptr, _MM_HINT_T0);
                        if self.dim > 16 {
                            _mm_prefetch(ptr.add(64), _MM_HINT_T0);
                        }
                    }
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = node_id;
        }
    }

    #[inline]
    fn compute_dist(&self, a: &[f32], b: &[f32], b_safe: bool) -> f64 {
        if self.safe_for_f32_dist && b_safe {
            l2_distance_squared_f32_fast(a, b) as f64
        } else {
            l2_distance_squared_unchecked(a, b)
        }
    }

    #[inline]
    fn dist_to_query(&mut self, node_id: u32, query: &[f32], query_safe: bool) -> Result<f64> {
        if !self.spilled {
            let start = (node_id as usize) * self.dim;
            return Ok(self.compute_dist(
                &self.in_mem[start..start + self.dim],
                query,
                query_safe,
            ));
        }
        if let Some((v, _)) = self.lru_map.get(&node_id) {
            return Ok(self.compute_dist(v, query, query_safe));
        }
        let v = self.read_from_spill(node_id)?;
        let d = self.compute_dist(&v, query, query_safe);
        self.insert_lru(node_id, v);
        Ok(d)
    }

    fn read_from_spill(&self, node_id: u32) -> Result<Vec<f32>> {
        let file = self.spill_file.as_ref().ok_or_else(|| {
            DbError::new(ErrorCode::InternalError, "spill file missing in HNSW")
        })?;
        let offset = (node_id as u64) * (self.dim as u64) * 4;
        let mut raw = vec![0u8; self.dim * 4];
        file.read_exact_at(&mut raw, offset)?;
        let mut out = Vec::with_capacity(self.dim);
        for chunk in raw.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    #[inline]
    fn dist_between(&mut self, a: u32, b: u32) -> Result<f64> {
        if !self.spilled {
            let sa = (a as usize) * self.dim;
            let sb = (b as usize) * self.dim;
            let va = &self.in_mem[sa..sa + self.dim];
            let vb = &self.in_mem[sb..sb + self.dim];
            if self.safe_for_f32_dist {
                return Ok(l2_distance_squared_f32_fast(va, vb) as f64);
            } else {
                return Ok(l2_distance_squared_unchecked(va, vb));
            }
        }
        let va = self.get_owned(a)?;
        let va_safe = self.safe_for_f32_dist && is_slice_safe_for_f32(&va);
        self.dist_to_query(b, &va, va_safe)
    }

    fn get_owned(&mut self, node_id: u32) -> Result<Vec<f32>> {
        if !self.spilled {
            let start = (node_id as usize) * self.dim;
            return Ok(self.in_mem[start..start + self.dim].to_vec());
        }
        if let Some((v, _)) = self.lru_map.get(&node_id) {
            return Ok(v.clone());
        }
        let v = self.read_from_spill(node_id)?;
        self.insert_lru(node_id, v.clone());
        Ok(v)
    }
}

#[derive(Debug, Clone, Copy)]
struct Neighbor {
    id: u32,
    /// `>= 0.0`: Tier-1 diverse neighbor with `dist = signed_dist`.
    /// `< 0.0` (and `!= NEG_INFINITY`): Tier-2 padding neighbor with `dist = -signed_dist - 1.0`.
    /// `NEG_INFINITY`: lazy uninitialized distance after `load_snapshot`.
    signed_dist: f32,
}

impl Neighbor {
    #[inline]
    fn new(id: u32, dist: f32, diverse: bool) -> Self {
        let d = dist.max(0.0);
        Self {
            id,
            signed_dist: if diverse { d } else { -(d + 1.0) },
        }
    }

    #[inline]
    fn uninit(id: u32) -> Self {
        Self {
            id,
            signed_dist: f32::NEG_INFINITY,
        }
    }

    #[inline]
    fn is_uninit(self) -> bool {
        self.signed_dist == f32::NEG_INFINITY
    }

    #[inline]
    fn is_diverse(self) -> bool {
        self.signed_dist >= 0.0
    }

    #[inline]
    fn dist(self) -> f32 {
        if self.signed_dist >= 0.0 {
            self.signed_dist
        } else {
            -self.signed_dist - 1.0
        }
    }

    #[inline]
    fn set_diverse(&mut self, diverse: bool) {
        let d = self.dist();
        self.signed_dist = if diverse { d } else { -(d + 1.0) };
    }
}

#[derive(Debug, Clone)]
struct HnswNode {
    row_id: RowId,
    pk: Option<i64>,
    level: u8,
    deleted: bool,
    neighbors: Vec<Vec<Neighbor>>,
}

/// Autonomous HNSW approximate nearest neighbor index with bounded memory,
/// transaction-safe incremental maintenance, filtered search, and snapshot persistence.
#[derive(Debug)]
pub struct HnswIndex {
    pub index_name: String,
    pub dim: usize,
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    m_l: f64,
    nodes: Vec<HnswNode>,
    row_to_node: HashMap<RowId, u32>,
    entry_point: Option<u32>,
    max_level: usize,
    live_count: usize,
    vectors: VectorStore,
    visited: Vec<u32>,
    visit_gen: u32,
    dirty: bool,
    needs_table_vectors: bool,
}

impl HnswIndex {
    pub fn new(index_name: String, dim: usize, spill_path: PathBuf) -> Self {
        Self::with_cache_limit(index_name, dim, spill_path, MAX_INMEM_VECTOR_BYTES)
    }

    pub fn with_cache_limit(
        index_name: String,
        dim: usize,
        spill_path: PathBuf,
        max_cache_bytes: usize,
    ) -> Self {
        let m = 16;
        Self {
            index_name,
            dim,
            m,
            m0: m * 2,
            ef_construction: 160,
            ef_search: 256,
            m_l: 1.0 / (m as f64).ln(),
            nodes: Vec::new(),
            row_to_node: HashMap::new(),
            entry_point: None,
            max_level: 0,
            live_count: 0,
            vectors: VectorStore::with_limit(dim, spill_path, max_cache_bytes),
            visited: Vec::new(),
            visit_gen: 0,
            dirty: true,
            needs_table_vectors: false,
        }
    }

    pub fn is_spilled(&self) -> bool {
        self.vectors.spilled
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    pub fn is_safe_for_f32_dist(&self) -> bool {
        self.vectors.safe_for_f32_dist
    }

    pub fn needs_table_vectors(&self) -> bool {
        self.needs_table_vectors
    }

    /// Scan all live nodes directly from the flat in-memory vector buffer (`!spilled && !needs_table_vectors`).
    pub fn scan_in_mem_live<F>(&self, mut f: F) -> Result<bool>
    where
        F: FnMut(RowId, Option<i64>, &[f32]) -> Result<()>,
    {
        if self.vectors.spilled || self.needs_table_vectors {
            return Ok(false);
        }
        let dim = self.dim;
        let in_mem = &self.vectors.in_mem;
        for (nid, node) in self.nodes.iter().enumerate() {
            if node.deleted {
                continue;
            }
            let start = nid * dim;
            f(node.row_id, node.pk, &in_mem[start..start + dim])?;
        }
        Ok(true)
    }

    pub fn live_count(&self) -> usize {
        self.live_count
    }

    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn needs_rebuild(&self) -> bool {
        self.nodes.len() > 512 && self.live_count * 2 < self.nodes.len()
    }

    pub fn reserve(&mut self, n_nodes: usize) {
        self.nodes.reserve_exact(n_nodes);
        self.row_to_node.reserve(n_nodes);
        self.visited.reserve_exact(n_nodes);
        if !self.vectors.spilled {
            let needed_floats = n_nodes.saturating_mul(self.dim);
            if needed_floats * std::mem::size_of::<f32>() <= self.vectors.max_inmem_bytes {
                self.vectors.in_mem.reserve_exact(needed_floats);
            }
        }
    }

    /// Free in-memory vectors and temporary buffers right before a table-backed (`spilled = 2`) shutdown snapshot.
    pub fn release_inmem_vectors_for_shutdown(&mut self) {
        if !self.vectors.spilled {
            self.vectors.in_mem = Vec::new();
            self.vectors.in_mem.shrink_to_fit();
        }
        self.visited = Vec::new();
        self.visited.shrink_to_fit();
    }

    fn next_visit_gen(&mut self) -> u32 {
        if self.visited.len() < self.nodes.len() {
            self.visited.resize(self.nodes.len(), 0);
        }
        if self.visit_gen == u32::MAX {
            self.visited.fill(0);
            self.visit_gen = 1;
        } else {
            self.visit_gen += 1;
        }
        self.visit_gen
    }

    #[inline]
    fn random_level(&self, row_id: RowId, node_seq: u32) -> usize {
        let mut x = row_id ^ ((node_seq as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        let u = ((x >> 11) as f64 + 0.5) / ((1u64 << 53) as f64);
        let lvl = (-u.ln() * self.m_l).floor() as usize;
        lvl.min(16)
    }

    pub fn delete(&mut self, row_id: RowId) {
        if let Some(nid) = self.row_to_node.remove(&row_id) {
            if !self.nodes[nid as usize].deleted {
                self.nodes[nid as usize].deleted = true;
                self.live_count = self.live_count.saturating_sub(1);
                self.dirty = true;
            }
        }
    }

    pub fn upsert(&mut self, row_id: RowId, pk: Option<i64>, vec: &[f32]) -> Result<()> {
        if vec.len() != self.dim {
            return Err(DbError::new(
                ErrorCode::DimensionMismatch,
                format!("HNSW upsert dim {} != {}", vec.len(), self.dim),
            ));
        }
        self.delete(row_id);
        self.dirty = true;

        let nid = u32::try_from(self.nodes.len())
            .map_err(|_| DbError::new(ErrorCode::InternalError, "HNSW node count overflow"))?;
        let level = self.random_level(row_id, nid);

        self.vectors.append(nid, vec)?;
        let vec_safe = self.vectors.safe_for_f32_dist;

        let mut neighbors = Vec::with_capacity(level + 1);
        for l in 0..=level {
            let cap = if l == 0 { self.m0 + 1 } else { self.m + 1 };
            neighbors.push(Vec::with_capacity(cap));
        }
        self.nodes.push(HnswNode {
            row_id,
            pk,
            level: level as u8,
            deleted: false,
            neighbors,
        });
        self.row_to_node.insert(row_id, nid);
        self.live_count += 1;

        let Some(mut cur_ep) = self.entry_point else {
            self.entry_point = Some(nid);
            self.max_level = level;
            return Ok(());
        };

        let mut cur_dist = self.vectors.dist_to_query(cur_ep, vec, vec_safe)?;
        if self.max_level > level {
            for l in (level + 1..=self.max_level).rev() {
                let (next_ep, next_dist) =
                    self.greedy_search_layer(vec, vec_safe, cur_ep, cur_dist, l)?;
                cur_ep = next_ep;
                cur_dist = next_dist;
            }
        }

        let top_connect = level.min(self.max_level);
        for l in (0..=top_connect).rev() {
            let candidates =
                self.search_layer(vec, vec_safe, cur_ep, cur_dist, self.ef_construction, l)?;
            if let Some(best) = candidates.first() {
                cur_ep = best.node_id;
                cur_dist = best.dist;
            }
            let max_links = if l == 0 { self.m0 } else { self.m };
            let selected = self.select_neighbors_heuristic(&candidates, max_links)?;
            self.nodes[nid as usize].neighbors[l] = selected.clone();

            for &nb in &selected {
                self.add_neighbor_heuristic_incremental(nb.id, l, nid, nb.dist(), max_links)?;
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(nid);
        }
        Ok(())
    }

    fn ensure_neighbor_distances_populated(&mut self, nb: u32, layer: usize) -> Result<()> {
        let needs_init = self.nodes[nb as usize].neighbors[layer]
            .iter()
            .any(|n| n.is_uninit());
        if !needs_init {
            return Ok(());
        }
        let links = std::mem::take(&mut self.nodes[nb as usize].neighbors[layer]);
        let mut cands = Vec::with_capacity(links.len());
        for item in links {
            let d = self.vectors.dist_between(nb, item.id)?;
            cands.push(DistNode {
                dist: d,
                node_id: item.id,
            });
        }
        cands.sort();
        let max_links = if layer == 0 { self.m0 } else { self.m };
        self.nodes[nb as usize].neighbors[layer] =
            self.select_neighbors_heuristic(&cands, max_links)?;
        Ok(())
    }

    fn add_neighbor_heuristic_incremental(
        &mut self,
        nb: u32,
        layer: usize,
        new_nid: u32,
        new_dist: f32,
        max_links: usize,
    ) -> Result<()> {
        self.ensure_neighbor_distances_populated(nb, layer)?;

        let cur_len = self.nodes[nb as usize].neighbors[layer].len();
        let ins_idx = self.nodes[nb as usize].neighbors[layer].partition_point(|n| {
            n.dist()
                .total_cmp(&new_dist)
                .then_with(|| n.id.cmp(&new_nid))
                == Ordering::Less
        });

        // Fast exit: if list is already full of Tier-1 neighbors and `new_nid` is farther than all of them,
        // it can never be retained even if diverse.
        if cur_len >= max_links
            && ins_idx >= max_links
            && self.nodes[nb as usize].neighbors[layer]
                .iter()
                .all(|n| n.is_diverse())
        {
            return Ok(());
        }

        // Test occlusion ONLY against closer Tier-1 (diverse) neighbors
        let mut new_diverse = true;
        for i in 0..ins_idx {
            let w = self.nodes[nb as usize].neighbors[layer][i];
            if w.is_diverse() {
                let d_uw = self.vectors.dist_between(new_nid, w.id)? as f32;
                if d_uw < new_dist {
                    new_diverse = false;
                    break;
                }
            }
        }

        if !new_diverse {
            if cur_len < max_links {
                self.nodes[nb as usize].neighbors[layer]
                    .insert(ins_idx, Neighbor::new(new_nid, new_dist, false));
                return Ok(());
            }
            // Find the farthest Tier-2 (!is_diverse) neighbor; replace it only if `new_dist` is closer
            if let Some(last_t2) = self.nodes[nb as usize].neighbors[layer]
                .iter()
                .rposition(|n| !n.is_diverse())
            {
                if new_dist < self.nodes[nb as usize].neighbors[layer][last_t2].dist() {
                    self.nodes[nb as usize].neighbors[layer].remove(last_t2);
                    let pos = self.nodes[nb as usize].neighbors[layer].partition_point(|n| {
                        n.dist()
                            .total_cmp(&new_dist)
                            .then_with(|| n.id.cmp(&new_nid))
                            == Ordering::Less
                    });
                    self.nodes[nb as usize].neighbors[layer]
                        .insert(pos, Neighbor::new(new_nid, new_dist, false));
                }
            }
            return Ok(());
        }

        // `new_nid` is a Tier-1 diverse neighbor; insert at `ins_idx`
        self.nodes[nb as usize].neighbors[layer]
            .insert(ins_idx, Neighbor::new(new_nid, new_dist, true));

        let new_len = self.nodes[nb as usize].neighbors[layer].len();
        let mut any_demoted = false;
        let mut promoted_ids: [u32; 32] = [0; 32];
        let mut promoted_len = 0usize;

        for i in (ins_idx + 1)..new_len {
            let item = self.nodes[nb as usize].neighbors[layer][i];
            let item_dist = item.dist();
            if item.is_diverse() {
                // Existing Tier-1 node: already non-occluded by all old Tier-1 nodes in `0..i`.
                // Only check against `new_nid` and any newly promoted nodes.
                let mut occluded =
                    (self.vectors.dist_between(item.id, new_nid)? as f32) < item_dist;
                if !occluded {
                    for &pid in &promoted_ids[..promoted_len] {
                        if (self.vectors.dist_between(item.id, pid)? as f32) < item_dist {
                            occluded = true;
                            break;
                        }
                    }
                }
                if occluded {
                    self.nodes[nb as usize].neighbors[layer][i].set_diverse(false);
                    any_demoted = true;
                }
            } else if any_demoted {
                // Existing Tier-2 node when an earlier Tier-1 node was demoted:
                // check `new_nid` first before scanning `0..i`.
                if (self.vectors.dist_between(item.id, new_nid)? as f32) < item_dist {
                    continue;
                }
                let mut occluded = false;
                for j in 0..i {
                    if j == ins_idx {
                        continue;
                    }
                    let prev = self.nodes[nb as usize].neighbors[layer][j];
                    if prev.is_diverse() {
                        let d = self.vectors.dist_between(item.id, prev.id)? as f32;
                        if d < item_dist {
                            occluded = true;
                            break;
                        }
                    }
                }
                if !occluded {
                    self.nodes[nb as usize].neighbors[layer][i].set_diverse(true);
                    if promoted_len < promoted_ids.len() {
                        promoted_ids[promoted_len] = item.id;
                        promoted_len += 1;
                    }
                }
            }
        }

        if new_len > max_links {
            if let Some(last_t2) = self.nodes[nb as usize].neighbors[layer]
                .iter()
                .rposition(|n| !n.is_diverse())
            {
                self.nodes[nb as usize].neighbors[layer].remove(last_t2);
            } else {
                self.nodes[nb as usize].neighbors[layer].pop();
            }
        }
        Ok(())
    }

    fn greedy_search_layer(
        &mut self,
        query: &[f32],
        query_safe: bool,
        mut cur_id: u32,
        mut cur_dist: f64,
        layer: usize,
    ) -> Result<(u32, f64)> {
        let nodes = &self.nodes;
        let vectors = &mut self.vectors;
        loop {
            let mut changed = false;
            for nb in &nodes[cur_id as usize].neighbors[layer] {
                let d = vectors.dist_to_query(nb.id, query, query_safe)?;
                if d < cur_dist {
                    cur_dist = d;
                    cur_id = nb.id;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Ok((cur_id, cur_dist))
    }

    /// Search a single layer returning candidates sorted ascending by distance.
    fn search_layer(
        &mut self,
        query: &[f32],
        query_safe: bool,
        ep_id: u32,
        ep_dist: f64,
        ef: usize,
        layer: usize,
    ) -> Result<Vec<DistNode>> {
        let vgen = self.next_visit_gen();
        self.visited[ep_id as usize] = vgen;

        let nodes = &self.nodes;
        let vectors = &mut self.vectors;
        let visited = &mut self.visited;

        let mut candidates: BinaryHeap<Reverse<DistNode>> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<DistNode> = BinaryHeap::with_capacity(ef + 1);
        let ep_entry = DistNode {
            dist: ep_dist,
            node_id: ep_id,
        };
        candidates.push(Reverse(ep_entry));
        results.push(ep_entry);

        while let Some(Reverse(cur)) = candidates.pop() {
            let worst_res = results.peek().map_or(f64::INFINITY, |x| x.dist);
            if cur.dist > worst_res && results.len() >= ef {
                break;
            }
            let nbs = &nodes[cur.node_id as usize].neighbors[layer];
            for (i, nb) in nbs.iter().enumerate() {
                let nb_id = nb.id;
                if visited[nb_id as usize] == vgen {
                    continue;
                }
                visited[nb_id as usize] = vgen;
                if i + 1 < nbs.len() {
                    vectors.prefetch_node(nbs[i + 1].id);
                }
                let d = vectors.dist_to_query(nb_id, query, query_safe)?;
                if results.len() < ef {
                    let item = DistNode {
                        dist: d,
                        node_id: nb_id,
                    };
                    candidates.push(Reverse(item));
                    results.push(item);
                } else if let Some(mut worst) = results.peek_mut() {
                    if d < worst.dist {
                        let item = DistNode {
                            dist: d,
                            node_id: nb_id,
                        };
                        *worst = item;
                        candidates.push(Reverse(item));
                    }
                }
            }
        }
        Ok(results.into_sorted_vec())
    }

    fn select_neighbors_heuristic(
        &mut self,
        sorted_candidates: &[DistNode],
        max_links: usize,
    ) -> Result<Vec<Neighbor>> {
        let mut selected: Vec<Neighbor> = Vec::with_capacity(sorted_candidates.len().min(max_links));
        let mut discarded: Vec<Neighbor> = Vec::new();

        for &cand in sorted_candidates {
            if selected.len() >= max_links {
                break;
            }
            let mut closer_to_existing = false;
            for sel in &selected {
                let inter_dist = self.vectors.dist_between(cand.node_id, sel.id)?;
                if inter_dist < cand.dist {
                    closer_to_existing = true;
                    break;
                }
            }
            if !closer_to_existing {
                selected.push(Neighbor::new(cand.node_id, cand.dist as f32, true));
            } else {
                discarded.push(Neighbor::new(cand.node_id, cand.dist as f32, false));
            }
        }

        if selected.len() < max_links && !discarded.is_empty() {
            for cand in discarded {
                if selected.len() >= max_links {
                    break;
                }
                selected.push(cand);
            }
            selected.sort_by(|a, b| a.dist().total_cmp(&b.dist()).then_with(|| a.id.cmp(&b.id)));
        }

        Ok(selected)
    }

    /// Approximate TopK search supporting deleted nodes, txn exclusions, and optional `WHERE` filter.
    /// Returns `(results_sorted_by_dist_and_pk, exhaustive_for_filter)`.
    pub fn search_knn<F>(
        &mut self,
        query: &[f32],
        k: usize,
        tx_excluded: &HashSet<RowId>,
        mut filter_fn: Option<&mut F>,
    ) -> Result<(Vec<(f64, Option<i64>, RowId)>, bool)>
    where
        F: FnMut(RowId) -> Result<bool>,
    {
        if k == 0 || self.live_count == 0 {
            return Ok((Vec::new(), true));
        }
        let Some(mut cur_ep) = self.entry_point else {
            return Ok((Vec::new(), true));
        };
        if query.len() != self.dim {
            return Err(DbError::new(
                ErrorCode::DimensionMismatch,
                format!("query dim {} != index dim {}", query.len(), self.dim),
            ));
        }

        let query_safe = is_slice_safe_for_f32(query);
        let mut cur_dist = self.vectors.dist_to_query(cur_ep, query, query_safe)?;
        for l in (1..=self.max_level).rev() {
            let (next_ep, next_dist) =
                self.greedy_search_layer(query, query_safe, cur_ep, cur_dist, l)?;
            cur_ep = next_ep;
            cur_dist = next_dist;
        }

        let is_filtered = filter_fn.is_some();
        let ef = if is_filtered {
            self.ef_search.max(k.saturating_mul(16))
        } else if self.dim >= 64 && self.nodes.len() >= 4096 {
            self.ef_search.max(272).max(k.saturating_mul(4))
        } else {
            self.ef_search.max(k.saturating_mul(4))
        };

        let vgen = self.next_visit_gen();
        self.visited[cur_ep as usize] = vgen;

        let nodes = &self.nodes;
        let vectors = &mut self.vectors;
        let visited = &mut self.visited;

        let mut candidates: BinaryHeap<Reverse<DistNode>> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<DistNode> = BinaryHeap::with_capacity(ef + 1);

        if !is_filtered && tx_excluded.is_empty() {
            candidates.push(Reverse(DistNode {
                dist: cur_dist,
                node_id: cur_ep,
            }));
            if !nodes[cur_ep as usize].deleted {
                results.push(DistNode {
                    dist: cur_dist,
                    node_id: cur_ep,
                });
            }
            while let Some(Reverse(cur)) = candidates.pop() {
                if results.len() >= ef {
                    if let Some(worst) = results.peek() {
                        if cur.dist > worst.dist {
                            break;
                        }
                    }
                }
                let nbs = &nodes[cur.node_id as usize].neighbors[0];
                for (i, nb) in nbs.iter().enumerate() {
                    let nb_id = nb.id;
                    if visited[nb_id as usize] == vgen {
                        continue;
                    }
                    visited[nb_id as usize] = vgen;
                    if i + 1 < nbs.len() {
                        vectors.prefetch_node(nbs[i + 1].id);
                    }
                    let d = vectors.dist_to_query(nb_id, query, query_safe)?;
                    if results.len() < ef {
                        let item = DistNode {
                            dist: d,
                            node_id: nb_id,
                        };
                        candidates.push(Reverse(item));
                        if !nodes[nb_id as usize].deleted {
                            results.push(item);
                        }
                    } else if let Some(mut worst) = results.peek_mut() {
                        if d < worst.dist {
                            let item = DistNode {
                                dist: d,
                                node_id: nb_id,
                            };
                            candidates.push(Reverse(item));
                            if !nodes[nb_id as usize].deleted {
                                *worst = item;
                            }
                        }
                    }
                }
            }
            let mut out: Vec<(f64, Option<i64>, RowId)> = results
                .into_iter()
                .map(|dn| {
                    let node = &self.nodes[dn.node_id as usize];
                    (dn.dist, node.pk, node.row_id)
                })
                .collect();
            out.sort_by(|a, b| {
                a.0.total_cmp(&b.0)
                    .then_with(|| a.1.cmp(&b.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            let exhaustive = out.len() >= k || out.len() >= self.live_count;
            out.truncate(k);
            return Ok((out, exhaustive));
        }

        let visit_budget = if is_filtered {
            (ef * 24).max(6144)
        } else {
            usize::MAX
        };
        let mut explore_pool: BinaryHeap<DistNode> = BinaryHeap::with_capacity(ef * 2 + 1);

        candidates.push(Reverse(DistNode {
            dist: cur_dist,
            node_id: cur_ep,
        }));
        explore_pool.push(DistNode {
            dist: cur_dist,
            node_id: cur_ep,
        });

        let mut visits = 1usize;
        let mut live_visited = 0usize;
        let mut filter_passed = 0usize;

        {
            let node = &nodes[cur_ep as usize];
            if !node.deleted && !tx_excluded.contains(&node.row_id) {
                live_visited += 1;
                let pass = match filter_fn.as_deref_mut() {
                    Some(f) => f(node.row_id)?,
                    None => true,
                };
                if pass {
                    filter_passed += 1;
                    results.push(DistNode {
                        dist: cur_dist,
                        node_id: cur_ep,
                    });
                }
            }
        }

        while let Some(Reverse(cur)) = candidates.pop() {
            if !is_filtered {
                let worst_res = results.peek().map_or(f64::INFINITY, |x| x.dist);
                if results.len() >= ef && cur.dist > worst_res {
                    break;
                }
            } else {
                let worst_res = results.peek().map_or(f64::INFINITY, |x| x.dist);
                if results.len() >= ef && cur.dist > worst_res {
                    break;
                }
                if visits >= visit_budget {
                    break;
                }
                // Adaptive bail-out for low selectivity filters (< ~12.5%):
                // hand off early to exact filter-first scan instead of scanning thousands of graph nodes.
                if live_visited >= 96 && filter_passed * 8 < live_visited {
                    return Ok((Vec::new(), false));
                }
            }

            let nbs = &nodes[cur.node_id as usize].neighbors[0];
            for (i, nb) in nbs.iter().enumerate() {
                let nb_id = nb.id;
                if visited[nb_id as usize] == vgen {
                    continue;
                }
                visited[nb_id as usize] = vgen;
                visits += 1;
                if i + 1 < nbs.len() {
                    vectors.prefetch_node(nbs[i + 1].id);
                }

                let d = vectors.dist_to_query(nb_id, query, query_safe)?;
                let worst_explore = explore_pool.peek().map_or(f64::INFINITY, |x| x.dist);
                let explore_cap = if is_filtered { (ef * 3).max(768) } else { ef };

                if explore_pool.len() < explore_cap
                    || d < worst_explore
                    || (is_filtered && results.len() < k)
                {
                    let item = DistNode {
                        dist: d,
                        node_id: nb_id,
                    };
                    candidates.push(Reverse(item));
                    explore_pool.push(item);
                    if explore_pool.len() > explore_cap {
                        explore_pool.pop();
                    }
                }

                let node = &nodes[nb_id as usize];
                if !node.deleted && !tx_excluded.contains(&node.row_id) {
                    live_visited += 1;
                    let worst_res = results.peek().map_or(f64::INFINITY, |x| x.dist);
                    if results.len() < ef || d < worst_res {
                        let pass = match filter_fn.as_deref_mut() {
                            Some(f) => f(node.row_id)?,
                            None => true,
                        };
                        if pass {
                            filter_passed += 1;
                            results.push(DistNode {
                                dist: d,
                                node_id: nb_id,
                            });
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<(f64, Option<i64>, RowId)> = results
            .into_iter()
            .map(|dn| {
                let node = &self.nodes[dn.node_id as usize];
                (dn.dist, node.pk, node.row_id)
            })
            .collect();

        out.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        let exhaustive = live_visited >= self.live_count || (!is_filtered && out.len() >= k);
        out.truncate(k);
        Ok((out, exhaustive))
    }

    pub fn save_snapshot(&mut self, snap_path: &Path, tmp_path: &Path) -> Result<()> {
        self.save_snapshot_internal(snap_path, tmp_path, false)
    }

    /// Save snapshot for a table-backed index (`spilled = 2` when vectors are in `DiskTable`,
    /// avoiding duplicate storage of raw vectors already persisted in `.dat`).
    pub fn save_snapshot_for_table(&mut self, snap_path: &Path, tmp_path: &Path) -> Result<()> {
        self.save_snapshot_internal(snap_path, tmp_path, true)
    }

    fn save_snapshot_internal(
        &mut self,
        snap_path: &Path,
        tmp_path: &Path,
        external_table_vectors: bool,
    ) -> Result<()> {
        if let Some(parent) = tmp_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if let Some(parent) = snap_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let compact_nbs = self.nodes.len() <= (u16::MAX as usize);
        let storage_mode: u8 = if self.vectors.spilled {
            1
        } else if external_table_vectors {
            2
        } else {
            0
        };
        let mode_byte = storage_mode | if compact_nbs { COMPACT_NEIGHBORS_FLAG } else { 0 };

        {
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(tmp_path)?;
            let mut w = CrcWriter::new(file);
            w.write_bytes(&HNSW_MAGIC)?;
            w.write_u32(1)?; // version
            w.write_u32(self.dim as u32)?;
            w.write_u32(self.m as u32)?;
            w.write_u32(self.m0 as u32)?;
            w.write_u32(self.ef_construction as u32)?;
            w.write_u32(self.max_level as u32)?;
            match self.entry_point {
                Some(ep) => {
                    w.write_u8(1)?;
                    w.write_u32(ep)?;
                }
                None => {
                    w.write_u8(0)?;
                    w.write_u32(0)?;
                }
            }
            w.write_u32(self.nodes.len() as u32)?;
            w.write_u32(self.live_count as u32)?;
            w.write_u8(mode_byte)?;

            for node in &self.nodes {
                w.write_u64(node.row_id)?;
                match node.pk {
                    Some(pk) => {
                        w.write_u8(1)?;
                        w.write_i64(pk)?;
                    }
                    None => {
                        w.write_u8(0)?;
                        w.write_i64(0)?;
                    }
                }
                w.write_u8(node.level)?;
                w.write_u8(if node.deleted { 1 } else { 0 })?;
                w.write_u8(node.neighbors.len() as u8)?;
                if compact_nbs {
                    for layer_nbs in &node.neighbors {
                        w.write_u8(layer_nbs.len() as u8)?;
                        for nb in layer_nbs {
                            w.write_u16(nb.id as u16)?;
                        }
                    }
                } else {
                    for layer_nbs in &node.neighbors {
                        w.write_u32(layer_nbs.len() as u32)?;
                        for nb in layer_nbs {
                            w.write_u32(nb.id)?;
                        }
                    }
                }
            }

            if storage_mode == 0 {
                w.write_u64(self.vectors.in_mem.len() as u64)?;
                for &x in &self.vectors.in_mem {
                    w.write_bytes(&x.to_le_bytes())?;
                }
            } else {
                if self.vectors.spilled {
                    if let Some(f) = self.vectors.spill_file.as_ref() {
                        f.sync_all()?;
                    }
                }
                w.write_u64(0)?;
            }

            let file = w.finish()?;
            file.sync_all()?;
        }

        fs::rename(tmp_path, snap_path)?;
        if let Some(parent) = snap_path.parent() {
            sync_dir(parent)?;
        }
        self.dirty = false;
        Ok(())
    }

    pub fn load_snapshot(
        index_name: String,
        snap_path: &Path,
        spill_path: PathBuf,
    ) -> Result<Option<Self>> {
        if !snap_path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(snap_path)?;
        if bytes.len() < 40 {
            return Ok(None);
        }
        let body_len = bytes.len() - 4;
        let expected_crc = u32::from_le_bytes(bytes[body_len..].try_into().unwrap());
        if crc32fast::hash(&bytes[..body_len]) != expected_crc {
            return Ok(None);
        }
        let mut cur = Cursor::new(&bytes[..body_len]);
        if cur.read_bytes(4)? != HNSW_MAGIC {
            return Ok(None);
        }
        let ver = cur.read_u32()?;
        if ver != 1 {
            return Ok(None);
        }
        let dim = cur.read_u32()? as usize;
        let m = cur.read_u32()? as usize;
        let m0 = cur.read_u32()? as usize;
        let ef_construction = cur.read_u32()? as usize;
        let max_level = cur.read_u32()? as usize;
        let has_ep = cur.read_u8()? != 0;
        let ep_raw = cur.read_u32()?;
        let entry_point = if has_ep { Some(ep_raw) } else { None };
        let node_count = cur.read_u32()? as usize;
        let live_count = cur.read_u32()? as usize;
        let mode_byte = cur.read_u8()?;
        let compact_nbs = (mode_byte & COMPACT_NEIGHBORS_FLAG) != 0;
        let storage_mode = mode_byte & !COMPACT_NEIGHBORS_FLAG;

        let mut nodes = Vec::with_capacity(node_count);
        let mut row_to_node = HashMap::with_capacity(live_count);
        for nid in 0..node_count {
            let row_id = cur.read_u64()?;
            let has_pk = cur.read_u8()? != 0;
            let pk_raw = cur.read_i64()?;
            let pk = if has_pk { Some(pk_raw) } else { None };
            let level = cur.read_u8()?;
            let deleted = cur.read_u8()? != 0;
            let nlayers = cur.read_u8()? as usize;
            let mut neighbors = Vec::with_capacity(nlayers);
            if compact_nbs {
                for _ in 0..nlayers {
                    let cnt = cur.read_u8()? as usize;
                    let mut lvec = Vec::with_capacity(cnt);
                    for _ in 0..cnt {
                        lvec.push(Neighbor::uninit(cur.read_u16()? as u32));
                    }
                    neighbors.push(lvec);
                }
            } else {
                for _ in 0..nlayers {
                    let cnt = cur.read_u32()? as usize;
                    let mut lvec = Vec::with_capacity(cnt);
                    for _ in 0..cnt {
                        lvec.push(Neighbor::uninit(cur.read_u32()?));
                    }
                    neighbors.push(lvec);
                }
            }
            if !deleted {
                row_to_node.insert(row_id, nid as u32);
            }
            nodes.push(HnswNode {
                row_id,
                pk,
                level,
                deleted,
                neighbors,
            });
        }

        let mut vectors = VectorStore::new(dim, spill_path.clone());
        let vec_len = cur.read_u64()? as usize;
        let mut needs_table_vectors = false;
        if storage_mode == 0 {
            let raw = cur.read_bytes(vec_len * 4)?;
            let mut in_mem = Vec::with_capacity(vec_len);
            let mut safe_f32 = true;
            for chunk in raw.chunks_exact(4) {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if safe_f32 && val.abs() >= 1e15 {
                    safe_f32 = false;
                }
                in_mem.push(val);
            }
            vectors.in_mem = in_mem;
            vectors.spilled = false;
            vectors.safe_for_f32_dist = safe_f32;
        } else if storage_mode == 1 {
            if !spill_path.exists() {
                return Ok(None);
            }
            let file = OpenOptions::new().read(true).write(true).open(&spill_path)?;
            if file.metadata()?.len() < (node_count as u64) * (dim as u64) * 4 {
                return Ok(None);
            }
            vectors.spill_file = Some(file);
            vectors.spilled = true;
        } else if storage_mode == 2 {
            needs_table_vectors = true;
            vectors.spilled = false;
        } else {
            return Ok(None);
        }

        Ok(Some(Self {
            index_name,
            dim,
            m,
            m0,
            ef_construction,
            ef_search: 128,
            m_l: 1.0 / (m as f64).ln(),
            nodes,
            row_to_node,
            entry_point,
            max_level,
            live_count,
            vectors,
            visited: Vec::new(),
            visit_gen: 0,
            dirty: false,
            needs_table_vectors,
        }))
    }

    /// Populate `VectorStore::in_mem` from `DiskTable` when loaded from an external-vector (`spilled = 2`) snapshot.
    pub fn populate_vectors_from_table(&mut self, dt: &DiskTable, col_idx: usize) -> Result<()> {
        if !self.needs_table_vectors {
            return Ok(());
        }
        let total_floats = self.nodes.len() * self.dim;
        let mut in_mem = vec![0.0_f32; total_floats];
        let mut safe_f32 = true;

        if dt.cache_covers_all() {
            for (nid, node) in self.nodes.iter().enumerate() {
                if node.deleted {
                    continue;
                }
                if let Some(row) = dt.cache.get(node.row_id) {
                    if let Some(Value::Vector(vec)) = row.values.get(col_idx) {
                        if safe_f32 && !is_slice_safe_for_f32(vec) {
                            safe_f32 = false;
                        }
                        let start = nid * self.dim;
                        in_mem[start..start + self.dim].copy_from_slice(vec);
                    }
                }
            }
        } else {
            for (nid, node) in self.nodes.iter().enumerate() {
                if node.deleted {
                    continue;
                }
                if let Some(vec) = dt.read_vector_col(node.row_id, col_idx)? {
                    if safe_f32 && !is_slice_safe_for_f32(&vec) {
                        safe_f32 = false;
                    }
                    let start = nid * self.dim;
                    in_mem[start..start + self.dim].copy_from_slice(&vec);
                }
            }
        }

        self.vectors.in_mem = in_mem;
        self.vectors.safe_for_f32_dist = safe_f32;
        self.needs_table_vectors = false;
        self.dirty = false;
        Ok(())
    }
}
