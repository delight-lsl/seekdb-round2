use crate::common::error::{DbError, ErrorCode, Result};
use crate::common::types::RowId;
use crate::storage::codec::{write_i64, write_u32, write_u64, write_u8, Cursor};
use crate::storage::file::atomic_write_file;
use crate::vector::distance::l2_distance_squared_unchecked;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

const HNSW_MAGIC: [u8; 4] = *b"HNS1";
const MAX_INMEM_VECTOR_BYTES: usize = 512 * 1024 * 1024; // 512 MiB heap budget for HNSW vectors

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
    lru_map: HashMap<u32, Vec<f32>>,
    lru_order: VecDeque<u32>,
    max_cached_nodes: usize,
    max_inmem_bytes: usize,
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
            max_cached_nodes,
            max_inmem_bytes,
        }
    }

    fn append(&mut self, node_id: u32, vec: &[f32]) -> Result<()> {
        debug_assert_eq!(vec.len(), self.dim);
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
        let mut writer = std::io::BufWriter::with_capacity(256 * 1024, &file);
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
            self.lru_map.insert(node_id, vec);
            return;
        }
        while self.lru_map.len() >= self.max_cached_nodes {
            let Some(evict) = self.lru_order.pop_front() else {
                break;
            };
            self.lru_map.remove(&evict);
        }
        self.lru_order.push_back(node_id);
        self.lru_map.insert(node_id, vec);
    }

    #[inline]
    fn dist_to_query(&mut self, node_id: u32, query: &[f32]) -> Result<f64> {
        if !self.spilled {
            let start = (node_id as usize) * self.dim;
            return Ok(l2_distance_squared_unchecked(
                &self.in_mem[start..start + self.dim],
                query,
            ));
        }
        if let Some(v) = self.lru_map.get(&node_id) {
            return Ok(l2_distance_squared_unchecked(v, query));
        }
        let v = self.read_from_spill(node_id)?;
        let d = l2_distance_squared_unchecked(&v, query);
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
            return Ok(l2_distance_squared_unchecked(
                &self.in_mem[sa..sa + self.dim],
                &self.in_mem[sb..sb + self.dim],
            ));
        }
        let va = self.get_owned(a)?;
        self.dist_to_query(b, &va)
    }

    fn get_owned(&mut self, node_id: u32) -> Result<Vec<f32>> {
        if !self.spilled {
            let start = (node_id as usize) * self.dim;
            return Ok(self.in_mem[start..start + self.dim].to_vec());
        }
        if let Some(v) = self.lru_map.get(&node_id) {
            return Ok(v.clone());
        }
        let v = self.read_from_spill(node_id)?;
        self.insert_lru(node_id, v.clone());
        Ok(v)
    }
}

#[derive(Debug, Clone)]
struct HnswNode {
    row_id: RowId,
    pk: Option<i64>,
    level: u8,
    deleted: bool,
    neighbors: Vec<Vec<u32>>,
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
        let m = 20;
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
        }
    }

    pub fn is_spilled(&self) -> bool {
        self.vectors.spilled
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
        // Uniform (0, 1] float
        let u = ((x >> 11) as f64 + 0.5) / ((1u64 << 53) as f64);
        let lvl = (-u.ln() * self.m_l).floor() as usize;
        lvl.min(16)
    }

    pub fn delete(&mut self, row_id: RowId) {
        if let Some(nid) = self.row_to_node.remove(&row_id) {
            if !self.nodes[nid as usize].deleted {
                self.nodes[nid as usize].deleted = true;
                self.live_count = self.live_count.saturating_sub(1);
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

        let nid = u32::try_from(self.nodes.len())
            .map_err(|_| DbError::new(ErrorCode::InternalError, "HNSW node count overflow"))?;
        let level = self.random_level(row_id, nid);

        self.vectors.append(nid, vec)?;
        self.nodes.push(HnswNode {
            row_id,
            pk,
            level: level as u8,
            deleted: false,
            neighbors: vec![Vec::new(); level + 1],
        });
        self.row_to_node.insert(row_id, nid);
        self.live_count += 1;

        let Some(mut cur_ep) = self.entry_point else {
            self.entry_point = Some(nid);
            self.max_level = level;
            return Ok(());
        };

        let mut cur_dist = self.vectors.dist_to_query(cur_ep, vec)?;
        if self.max_level > level {
            for l in (level + 1..=self.max_level).rev() {
                let (next_ep, next_dist) = self.greedy_search_layer(vec, cur_ep, cur_dist, l)?;
                cur_ep = next_ep;
                cur_dist = next_dist;
            }
        }

        let top_connect = level.min(self.max_level);
        for l in (0..=top_connect).rev() {
            let candidates = self.search_layer(vec, cur_ep, cur_dist, self.ef_construction, l)?;
            if let Some(best) = candidates.first() {
                cur_ep = best.node_id;
                cur_dist = best.dist;
            }
            let max_links = if l == 0 { self.m0 } else { self.m };
            let selected = self.select_neighbors_heuristic(&candidates, max_links)?;
            self.nodes[nid as usize].neighbors[l] = selected.clone();

            for &nb in &selected {
                self.nodes[nb as usize].neighbors[l].push(nid);
                if self.nodes[nb as usize].neighbors[l].len() > max_links {
                    let nb_links = std::mem::take(&mut self.nodes[nb as usize].neighbors[l]);
                    let mut nb_cands = Vec::with_capacity(nb_links.len());
                    for link_id in nb_links {
                        let d = self.vectors.dist_between(nb, link_id)?;
                        nb_cands.push(DistNode {
                            dist: d,
                            node_id: link_id,
                        });
                    }
                    nb_cands.sort();
                    let pruned = self.select_neighbors_heuristic(&nb_cands, max_links)?;
                    self.nodes[nb as usize].neighbors[l] = pruned;
                }
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(nid);
        }
        Ok(())
    }

    fn greedy_search_layer(
        &mut self,
        query: &[f32],
        mut cur_id: u32,
        mut cur_dist: f64,
        layer: usize,
    ) -> Result<(u32, f64)> {
        let nodes = &self.nodes;
        let vectors = &mut self.vectors;
        loop {
            let mut changed = false;
            for &nb in &nodes[cur_id as usize].neighbors[layer] {
                let d = vectors.dist_to_query(nb, query)?;
                if d < cur_dist {
                    cur_dist = d;
                    cur_id = nb;
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
            for &nb in &nodes[cur.node_id as usize].neighbors[layer] {
                if visited[nb as usize] == vgen {
                    continue;
                }
                visited[nb as usize] = vgen;
                let d = vectors.dist_to_query(nb, query)?;
                let worst_now = results.peek().map_or(f64::INFINITY, |x| x.dist);
                if results.len() < ef || d < worst_now {
                    let item = DistNode {
                        dist: d,
                        node_id: nb,
                    };
                    candidates.push(Reverse(item));
                    results.push(item);
                    if results.len() > ef {
                        results.pop();
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
    ) -> Result<Vec<u32>> {
        if sorted_candidates.len() <= max_links {
            return Ok(sorted_candidates.iter().map(|c| c.node_id).collect());
        }
        let mut selected: Vec<DistNode> = Vec::with_capacity(max_links);
        let mut discarded: Vec<DistNode> = Vec::new();

        for &cand in sorted_candidates {
            if selected.len() >= max_links {
                break;
            }
            let mut closer_to_existing = false;
            for sel in &selected {
                let inter_dist = self.vectors.dist_between(cand.node_id, sel.node_id)?;
                if inter_dist < cand.dist {
                    closer_to_existing = true;
                    break;
                }
            }
            if !closer_to_existing {
                selected.push(cand);
            } else {
                discarded.push(cand);
            }
        }

        for cand in discarded {
            if selected.len() >= max_links {
                break;
            }
            selected.push(cand);
        }

        Ok(selected.into_iter().map(|c| c.node_id).collect())
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

        let mut cur_dist = self.vectors.dist_to_query(cur_ep, query)?;
        for l in (1..=self.max_level).rev() {
            let (next_ep, next_dist) = self.greedy_search_layer(query, cur_ep, cur_dist, l)?;
            cur_ep = next_ep;
            cur_dist = next_dist;
        }

        let is_filtered = filter_fn.is_some();
        let ef = if is_filtered {
            self.ef_search.max(k.saturating_mul(16))
        } else {
            self.ef_search.max(k.saturating_mul(4))
        };
        let visit_budget = if is_filtered {
            (ef * 24).max(6144)
        } else {
            usize::MAX
        };

        let vgen = self.next_visit_gen();
        self.visited[cur_ep as usize] = vgen;

        let nodes = &self.nodes;
        let vectors = &mut self.vectors;
        let visited = &mut self.visited;

        let mut candidates: BinaryHeap<Reverse<DistNode>> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<DistNode> = BinaryHeap::with_capacity(ef + 1);
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
                // Adaptive bail-out for very low selectivity filters (< ~12%):
                // hand off early to exact filter-first scan instead of scanning thousands of graph nodes.
                if live_visited >= 256 && filter_passed * 8 < live_visited {
                    return Ok((Vec::new(), false));
                }
            }

            for &nb in &nodes[cur.node_id as usize].neighbors[0] {
                if visited[nb as usize] == vgen {
                    continue;
                }
                visited[nb as usize] = vgen;
                visits += 1;

                let d = vectors.dist_to_query(nb, query)?;
                let worst_explore = explore_pool.peek().map_or(f64::INFINITY, |x| x.dist);
                let explore_cap = if is_filtered { (ef * 3).max(768) } else { ef };

                if explore_pool.len() < explore_cap
                    || d < worst_explore
                    || (is_filtered && results.len() < k)
                {
                    let item = DistNode {
                        dist: d,
                        node_id: nb,
                    };
                    candidates.push(Reverse(item));
                    explore_pool.push(item);
                    if explore_pool.len() > explore_cap {
                        explore_pool.pop();
                    }
                }

                let node = &nodes[nb as usize];
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
                                node_id: nb,
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
        let mut buf = Vec::new();
        buf.extend_from_slice(&HNSW_MAGIC);
        write_u32(&mut buf, 1); // version
        write_u32(&mut buf, self.dim as u32);
        write_u32(&mut buf, self.m as u32);
        write_u32(&mut buf, self.m0 as u32);
        write_u32(&mut buf, self.ef_construction as u32);
        write_u32(&mut buf, self.max_level as u32);
        match self.entry_point {
            Some(ep) => {
                write_u8(&mut buf, 1);
                write_u32(&mut buf, ep);
            }
            None => {
                write_u8(&mut buf, 0);
                write_u32(&mut buf, 0);
            }
        }
        write_u32(&mut buf, self.nodes.len() as u32);
        write_u32(&mut buf, self.live_count as u32);
        write_u8(&mut buf, if self.vectors.spilled { 1 } else { 0 });

        for node in &self.nodes {
            write_u64(&mut buf, node.row_id);
            match node.pk {
                Some(pk) => {
                    write_u8(&mut buf, 1);
                    write_i64(&mut buf, pk);
                }
                None => {
                    write_u8(&mut buf, 0);
                    write_i64(&mut buf, 0);
                }
            }
            write_u8(&mut buf, node.level);
            write_u8(&mut buf, if node.deleted { 1 } else { 0 });
            write_u8(&mut buf, node.neighbors.len() as u8);
            for layer_nbs in &node.neighbors {
                write_u32(&mut buf, layer_nbs.len() as u32);
                for &nb in layer_nbs {
                    write_u32(&mut buf, nb);
                }
            }
        }

        if !self.vectors.spilled {
            write_u64(&mut buf, self.vectors.in_mem.len() as u64);
            for &x in &self.vectors.in_mem {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        } else {
            if let Some(f) = self.vectors.spill_file.as_ref() {
                f.sync_all()?;
            }
            write_u64(&mut buf, 0);
        }

        let crc = crc32fast::hash(&buf);
        write_u32(&mut buf, crc);
        atomic_write_file(snap_path, tmp_path, &buf)
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
        let spilled = cur.read_u8()? != 0;

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
            for _ in 0..nlayers {
                let cnt = cur.read_u32()? as usize;
                let mut lvec = Vec::with_capacity(cnt);
                for _ in 0..cnt {
                    lvec.push(cur.read_u32()?);
                }
                neighbors.push(lvec);
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
        if !spilled {
            let raw = cur.read_bytes(vec_len * 4)?;
            let mut in_mem = Vec::with_capacity(vec_len);
            for chunk in raw.chunks_exact(4) {
                in_mem.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            vectors.in_mem = in_mem;
            vectors.spilled = false;
        } else {
            if !spill_path.exists() {
                return Ok(None);
            }
            let file = OpenOptions::new().read(true).write(true).open(&spill_path)?;
            if file.metadata()?.len() < (node_count as u64) * (dim as u64) * 4 {
                return Ok(None);
            }
            vectors.spill_file = Some(file);
            vectors.spilled = true;
        }

        Ok(Some(Self {
            index_name,
            dim,
            m,
            m0,
            ef_construction,
            ef_search: 256,
            m_l: 1.0 / (m as f64).ln(),
            nodes,
            row_to_node,
            entry_point,
            max_level,
            live_count,
            vectors,
            visited: Vec::new(),
            visit_gen: 0,
        }))
    }
}
