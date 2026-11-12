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
