use crate::common::types::{Row, RowId};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Candidate entry for bounded K-nearest TopK retrieval using a max-heap.
/// Ordering places the WORST candidate (largest squared distance, then largest PK/RowId) at the root,
/// allowing O(log K) eviction and O(K log K) final ascending output via `into_sorted_vec()`.
#[derive(Debug, Clone)]
pub struct TopKCandidate {
    pub dist_sq: f64,
    pub pk: Option<i64>,
    pub row_id: RowId,
    pub row: Row,
}

impl PartialEq for TopKCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq.to_bits() == other.dist_sq.to_bits()
            && self.pk == other.pk
            && self.row_id == other.row_id
    }
}

impl Eq for TopKCandidate {}

impl PartialOrd for TopKCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TopKCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist_sq
            .total_cmp(&other.dist_sq)
            .then_with(|| self.pk.cmp(&other.pk))
            .then_with(|| self.row_id.cmp(&other.row_id))
    }
}

/// Bounded max-heap of size K maintaining TopK nearest items in O(N log K) time and O(K) memory.
pub struct TopKHeap {
    k: usize,
    heap: BinaryHeap<TopKCandidate>,
}

impl TopKHeap {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.saturating_add(1)),
        }
    }

    #[inline]
    pub fn would_accept(&self, dist_sq: f64, pk: Option<i64>, row_id: RowId) -> bool {
        if self.k == 0 {
            return false;
        }
        if self.heap.len() < self.k {
            return true;
        }
        if let Some(worst) = self.heap.peek() {
            let ord = dist_sq
                .total_cmp(&worst.dist_sq)
                .then_with(|| pk.cmp(&worst.pk))
                .then_with(|| row_id.cmp(&worst.row_id));
            ord == Ordering::Less
        } else {
            true
        }
    }

    #[inline]
    pub fn push(&mut self, item: TopKCandidate) {
        if self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(item);
        } else if let Some(mut worst) = self.heap.peek_mut() {
            if item < *worst {
                *worst = item;
            }
        }
    }

    pub fn into_sorted_vec(self) -> Vec<TopKCandidate> {
        self.heap.into_sorted_vec()
    }
}
