use crate::common::error::{DbError, ErrorCode, Result};

/// Fast squared L2 distance for validated equal-length `f32` slices,
/// accumulating in `f64` to avoid intermediate square overflow while preserving `f32` storage semantics.
#[inline]
pub fn l2_distance_squared_unchecked(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() >= 16 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime check confirms AVX2 and FMA are supported on this CPU.
            return unsafe { l2_sq_avx2_fma(a, b) };
        }
    }
    l2_sq_unrolled(a, b)
}

#[inline]
fn l2_sq_unrolled(a: &[f32], b: &[f32]) -> f64 {
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;

    let mut chunks_a = a.chunks_exact(8);
    let mut chunks_b = b.chunks_exact(8);
    for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        let d0 = (ca[0] as f64) - (cb[0] as f64);
        let d1 = (ca[1] as f64) - (cb[1] as f64);
        let d2 = (ca[2] as f64) - (cb[2] as f64);
        let d3 = (ca[3] as f64) - (cb[3] as f64);
        let d4 = (ca[4] as f64) - (cb[4] as f64);
        let d5 = (ca[5] as f64) - (cb[5] as f64);
        let d6 = (ca[6] as f64) - (cb[6] as f64);
        let d7 = (ca[7] as f64) - (cb[7] as f64);
        s0 += d0 * d0 + d4 * d4;
        s1 += d1 * d1 + d5 * d5;
        s2 += d2 * d2 + d6 * d6;
        s3 += d3 * d3 + d7 * d7;
    }

    let mut rem_sum = 0.0_f64;
    for (&x, &y) in chunks_a.remainder().iter().zip(chunks_b.remainder().iter()) {
        let d = (x as f64) - (y as f64);
        rem_sum += d * d;
    }

    (s0 + s1) + (s2 + s3) + rem_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_sq_avx2_fma(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut idx = 0;
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();

    while idx + 8 <= len {
        let va0 = _mm_loadu_ps(a.as_ptr().add(idx));
        let vb0 = _mm_loadu_ps(b.as_ptr().add(idx));
        let da0 = _mm256_cvtps_pd(va0);
        let db0 = _mm256_cvtps_pd(vb0);
        let diff0 = _mm256_sub_pd(da0, db0);
        acc0 = _mm256_fmadd_pd(diff0, diff0, acc0);

        let va1 = _mm_loadu_ps(a.as_ptr().add(idx + 4));
        let vb1 = _mm_loadu_ps(b.as_ptr().add(idx + 4));
        let da1 = _mm256_cvtps_pd(va1);
        let db1 = _mm256_cvtps_pd(vb1);
        let diff1 = _mm256_sub_pd(da1, db1);
        acc1 = _mm256_fmadd_pd(diff1, diff1, acc1);

        idx += 8;
    }

    let combined = _mm256_add_pd(acc0, acc1);
    let hi = _mm256_extractf128_pd(combined, 1);
    let lo = _mm256_castpd256_pd128(combined);
    let sum2 = _mm_add_pd(lo, hi);
    let shuffled = _mm_permute_pd(sum2, 0b01);
    let sum1 = _mm_add_sd(sum2, shuffled);
    let mut sum = _mm_cvtsd_f64(sum1);

    while idx < len {
        let d = (*a.get_unchecked(idx) as f64) - (*b.get_unchecked(idx) as f64);
        sum += d * d;
        idx += 1;
    }
    sum
}

/// Fast pure-`f32` squared L2 distance for HNSW graph construction and search
/// when vector components are within safe `f32` squaring range (`|x| < 1e15`).
#[inline]
pub fn l2_distance_squared_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() >= 16 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime check confirms AVX2 and FMA are supported on this CPU.
            return unsafe { l2_sq_f32_avx2_fma(a, b) };
        }
    }
    l2_sq_f32_unrolled(a, b)
}

#[inline]
fn l2_sq_f32_unrolled(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;

    let mut chunks_a = a.chunks_exact(8);
    let mut chunks_b = b.chunks_exact(8);
    for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        let d0 = ca[0] - cb[0];
        let d1 = ca[1] - cb[1];
        let d2 = ca[2] - cb[2];
        let d3 = ca[3] - cb[3];
        let d4 = ca[4] - cb[4];
        let d5 = ca[5] - cb[5];
        let d6 = ca[6] - cb[6];
        let d7 = ca[7] - cb[7];
        s0 += d0 * d0 + d4 * d4;
        s1 += d1 * d1 + d5 * d5;
        s2 += d2 * d2 + d6 * d6;
        s3 += d3 * d3 + d7 * d7;
    }

    let mut rem_sum = 0.0_f32;
    for (&x, &y) in chunks_a.remainder().iter().zip(chunks_b.remainder().iter()) {
        let d = x - y;
        rem_sum += d * d;
    }

    (s0 + s1) + (s2 + s3) + rem_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_sq_f32_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut idx = 0;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let pa = a.as_ptr();
    let pb = b.as_ptr();

    while idx + 32 <= len {
        let va0 = _mm256_loadu_ps(pa.add(idx));
        let vb0 = _mm256_loadu_ps(pb.add(idx));
        let d0 = _mm256_sub_ps(va0, vb0);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);

        let va1 = _mm256_loadu_ps(pa.add(idx + 8));
        let vb1 = _mm256_loadu_ps(pb.add(idx + 8));
        let d1 = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);

        let va2 = _mm256_loadu_ps(pa.add(idx + 16));
        let vb2 = _mm256_loadu_ps(pb.add(idx + 16));
        let d2 = _mm256_sub_ps(va2, vb2);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);

        let va3 = _mm256_loadu_ps(pa.add(idx + 24));
        let vb3 = _mm256_loadu_ps(pb.add(idx + 24));
        let d3 = _mm256_sub_ps(va3, vb3);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);

        idx += 32;
    }

    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

    while idx + 8 <= len {
        let va = _mm256_loadu_ps(pa.add(idx));
        let vb = _mm256_loadu_ps(pb.add(idx));
        let d = _mm256_sub_ps(va, vb);
        acc0 = _mm256_fmadd_ps(d, d, acc0);
        idx += 8;
    }

    let hi128 = _mm256_extractf128_ps(acc0, 1);
    let lo128 = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, shuf64);
    let shuf32 = _mm_shuffle_ps(sum64, sum64, 0x55);
    let sum32 = _mm_add_ss(sum64, shuf32);
    let mut sum = _mm_cvtss_f32(sum32);

    while idx < len {
        let d = *a.get_unchecked(idx) - *b.get_unchecked(idx);
        sum += d * d;
        idx += 1;
    }
    sum
}

/// Compute squared Euclidean (L2) distance with dimension and finiteness validation.
#[inline]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(DbError::new(
            ErrorCode::DimensionMismatch,
            format!("vector dimension mismatch: {} vs {}", a.len(), b.len()),
        ));
    }
    let sum = l2_distance_squared_unchecked(a, b);
    if !sum.is_finite() {
        return Err(DbError::new(
            ErrorCode::InvalidVector,
            "non-finite distance result",
        ));
    }
    Ok(sum)
}

/// Compute Euclidean (L2) distance between two f32 slices.
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> Result<f64> {
    Ok(l2_distance_squared(a, b)?.sqrt())
}

/// Single-precision candidate scan over contiguous vector blocks.
///
/// Returns the slots of every row whose f32 squared distance is within the rounding
/// slack of the k-th smallest one, i.e. a superset of the exact top-k (ties included).
/// The caller re-ranks this small set with the exact kernel.
/// Callers guarantee all components are below 1e15 in magnitude (no f32 overflow).
pub fn scan_topk_f32<'a, I>(blocks: I, dim: usize, query: &[f32], k: usize, mask: Option<&[u8]>) -> Vec<u32>
where
    I: Iterator<Item = (usize, usize, &'a [f32])>,
{
    use std::collections::BinaryHeap;
    debug_assert_eq!(query.len(), dim);
    #[cfg(target_arch = "x86_64")]
    let simd = dim >= 8 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let simd = false;

    // Non-negative finite f32 values order exactly like their bit patterns.
    let mut heap: BinaryHeap<u32> = BinaryHeap::with_capacity(k.min(4096) + 1);
    let mut cands: Vec<(f32, u32)> = Vec::with_capacity(k.min(4096) * 4 + 16);
    let mut thr = f32::INFINITY;
    // Worst-case relative rounding error of the f32 kernel is below (dim/8 + 32) ulps;
    // three times that covers "f32 value vs exact value" on both sides of the comparison.
    let slack = 1.0 + 3.0 * (dim as f32 / 8.0 + 32.0) * f32::EPSILON;

    for (first_slot, rows, data) in blocks {
        for r in 0..rows {
            let slot = first_slot + r;
            if let Some(m) = mask {
                if m[slot] == 0 {
                    continue;
                }
            }
            let v = &data[r * dim..(r + 1) * dim];
            let d = if simd {
                #[cfg(target_arch = "x86_64")]
                {
                    // SAFETY: AVX2/FMA availability was checked above; slices have `dim` elements.
                    unsafe { l2_sq_f32_avx2_fma(v, query) }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    l2_sq_f32_unrolled(v, query)
                }
            } else {
                l2_sq_f32_unrolled(v, query)
            };
            if d <= thr {
                cands.push((d, slot as u32));
                if heap.len() < k {
                    heap.push(d.to_bits());
                    if heap.len() == k {
                        thr = f32::from_bits(*heap.peek().unwrap()) * slack + f32::MIN_POSITIVE;
                    }
                } else if d.to_bits() < *heap.peek().unwrap() {
                    *heap.peek_mut().unwrap() = d.to_bits();
                    thr = f32::from_bits(*heap.peek().unwrap()) * slack + f32::MIN_POSITIVE;
                }
                if cands.len() >= 8192 && cands.len() >= k.saturating_mul(8) {
                    cands.retain(|c| c.0 <= thr);
                }
            }
        }
    }
    cands.into_iter().filter(|c| c.0 <= thr).map(|c| c.1).collect()
}

/// Integer lower bounds for one block of 7-bit codes (`rows` rows of `stride` bytes,
/// `stride` a multiple of 32): for every selected row, `sum(max(0, |c - q| - 1)^2)` is
/// appended to `lbs`. With a mask, the slot number of each selected row is appended to
/// `slots` as well (without a mask the position in `lbs` is the slot).
pub fn sq_lower_bounds(
    codes: &[u8],
    stride: usize,
    rows: usize,
    q: &[u8],
    mask: Option<&[u8]>,
    first_slot: u32,
    lbs: &mut Vec<u32>,
    slots: &mut Vec<u32>,
) {
    assert!(stride % 32 == 0 && q.len() == stride && codes.len() >= rows * stride);
    if let Some(m) = mask {
        assert!(m.len() >= rows);
    }
    lbs.reserve(rows);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability checked; bounds asserted above; capacity reserved.
            unsafe { sq_lower_bounds_avx2(codes, stride, rows, q, mask, first_slot, lbs, slots) };
            return;
        }
    }
    for r in 0..rows {
        if let Some(m) = mask {
            if m[r] == 0 {
                continue;
            }
            slots.push(first_slot + r as u32);
        }
        let row = &codes[r * stride..(r + 1) * stride];
        let mut lb = 0u32;
        for (a, b) in row.iter().zip(q.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs().saturating_sub(1);
            lb += d * d;
        }
        lbs.push(lb);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn sq_lower_bounds_avx2(
    codes: &[u8],
    stride: usize,
    rows: usize,
    q: &[u8],
    mask: Option<&[u8]>,
    first_slot: u32,
    lbs: &mut Vec<u32>,
    slots: &mut Vec<u32>,
) {
    use std::arch::x86_64::*;
    let one = _mm256_set1_epi8(1);
    let ones16 = _mm256_set1_epi16(1);
    let qp = q.as_ptr();
    let base = codes.as_ptr();
    let words = stride / 32;

    #[inline(always)]
    unsafe fn row_lb(rp: *const u8, qp: *const u8, words: usize, one: __m256i, ones16: __m256i) -> u32 {
        let mut acc = _mm256_setzero_si256();
        for w in 0..words {
            let x = _mm256_loadu_si256(rp.add(w * 32) as *const __m256i);
            let y = _mm256_loadu_si256(qp.add(w * 32) as *const __m256i);
            // |x - y| - 1 saturated at 0; codes are 7-bit, so the value fits a signed byte
            // and each maddubs pair (<= 2 * 126^2) fits a signed 16-bit lane.
            let d = _mm256_subs_epu8(_mm256_or_si256(_mm256_subs_epu8(x, y), _mm256_subs_epu8(y, x)), one);
            let sq16 = _mm256_maddubs_epi16(d, d);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(sq16, ones16));
        }
        let s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x11));
        _mm_cvtsi128_si32(s) as u32
    }

    match mask {
        None => {
            let out = lbs.as_mut_ptr().add(lbs.len());
            for r in 0..rows {
                *out.add(r) = row_lb(base.add(r * stride), qp, words, one, ones16);
            }
            lbs.set_len(lbs.len() + rows);
        }
        Some(m) => {
            let zero = _mm256_setzero_si256();
            let mp = m.as_ptr();
            let mut r0 = 0usize;
            while r0 < rows {
                // Bitmap of selected rows among the next (up to) 32.
                let mut bits: u32 = if r0 + 32 <= rows {
                    !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(mp.add(r0) as *const __m256i), zero)) as u32)
                } else {
                    let mut b = 0u32;
                    for j in 0..rows - r0 {
                        b |= ((*mp.add(r0 + j) != 0) as u32) << j;
                    }
                    b
                };
                while bits != 0 {
                    let r = r0 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    lbs.push(row_lb(base.add(r * stride), qp, words, one, ones16));
                    slots.push(first_slot + r as u32);
                }
                r0 += 32;
            }
        }
    }
}

/// Visit every index `i` with `vals[i] <= thr`, in increasing order. The callback
/// returns the threshold to use from then on (it may only stay or shrink); indexes
/// reported right after a change may exceed the new threshold by a few positions, so
/// the callback re-checks when it matters. All values must be below `2^31`.
pub fn scan_le<F>(vals: &[u32], thr: u32, mut f: F)
where
    F: FnMut(usize) -> u32,
{
    let mut thr = thr.min(i32::MAX as u32);
    let mut i = 0usize;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability checked; loads stay inside `vals`.
            unsafe {
                use std::arch::x86_64::*;
                let p = vals.as_ptr();
                let mut tv = _mm256_set1_epi32(thr as i32);
                while i + 8 <= vals.len() {
                    let v = _mm256_loadu_si256(p.add(i) as *const __m256i);
                    let gt = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(v, tv))) as u32;
                    let mut hit = !gt & 0xff;
                    if hit != 0 {
                        while hit != 0 {
                            let j = hit.trailing_zeros() as usize;
                            hit &= hit - 1;
                            if *p.add(i + j) <= thr {
                                thr = f(i + j).min(thr);
                            }
                        }
                        tv = _mm256_set1_epi32(thr as i32);
                    }
                    i += 8;
                }
            }
        }
    }
    while i < vals.len() {
        if vals[i] <= thr {
            thr = f(i).min(thr);
        }
        i += 1;
    }
}
