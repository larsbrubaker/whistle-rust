//! The few-row path of the dense kernel (`kernels`): `x · wᵀ` for fewer input rows than the
//! panel path's row block, as the decoder's cached step projects (one row per token, the
//! vocabulary head included). For so few rows the panel path's scratch transpose of every weight
//! panel costs about as much as the multiply, so this path transposes in registers instead: it
//! loads 4 inputs of each of 4 weight rows, transposes that 4×4 block with shuffles, and adds
//! each input's products to one lane per output. The order is the panel path's, and the port's:
//! each output is the sequential sum from `-0.0` in input order, product rounded then added, no
//! FMA; the inputs past the last multiple of 4 and the outputs past the last group are finished
//! by the same scalar formula. So every path here is bit-identical to the panel path.

use super::Path;

/// Output groups of 4 computed together for one row. More groups give more independent
/// accumulators but read more weight rows at once, and a decoder step streams about 72 MB of
/// weights per token, more than the caches hold. Measured 2026-10-24 (Apple M5, release,
/// `few_row_timing`, best of 10): 64 distinct 512×512 single-row projections (64 MB) took
/// 2168 µs with 1 group (4 rows read together), 2674 µs with 2 and 3567 µs with 4, against
/// 3245 µs by transposed panels; in cache (one 512×512 matrix) 4 groups win, 16 vs 32 µs.
const GROUPS: usize = 1;

/// Finishes outputs `c0..c0 + 4G` of row `xr` on inputs `from..n` after the vector part, in
/// order, and stores them; `acc[g][j]` holds output `c0 + 4g + j`'s sum over `0..from`.
#[inline(always)]
fn finish<const G: usize>(
    mut acc: [[f32; 4]; G],
    xr: &[f32],
    w: &[f32],
    n: usize,
    c0: usize,
    from: usize,
    out: &mut [f32],
) {
    for (g, lanes) in acc.iter_mut().enumerate() {
        for (j, a) in lanes.iter_mut().enumerate() {
            let wr = &w[(c0 + 4 * g + j) * n..(c0 + 4 * g + j + 1) * n];
            for i in from..n {
                *a += xr[i] * wr[i];
            }
            out[c0 + 4 * g + j] = *a;
        }
    }
}

/// The portable order: outputs `c0..c0 + 4G` of `xr`, each the sequential sum.
#[inline(always)]
fn group_scalar<const G: usize>(xr: &[f32], w: &[f32], n: usize, c0: usize, out: &mut [f32]) {
    finish([[-0.0f32; 4]; G], xr, w, n, c0, 0, out);
}

// Index loops over the fixed-size register arrays: they unroll fully and pair groups with lanes.
#[allow(clippy::needless_range_loop)]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn group_neon<const G: usize>(xr: &[f32], w: &[f32], n: usize, c0: usize, out: &mut [f32]) {
    use core::arch::aarch64::*;
    assert!(xr.len() >= n && w.len() >= (c0 + 4 * G) * n);
    let n4 = n - n % 4;
    let mut lanes = [[0.0f32; 4]; G];
    // SAFETY: NEON is part of the aarch64 baseline. Weight row `c0 + 4g + j` (`< c0 + 4G`) is
    // read as 4 floats at input `i < n4 ≤ n`, within `w` (asserted `(c0 + 4G) · n` long); the
    // input loads `xr[i..i + 4]` are within `xr` (asserted `n` long); the stores write a local
    // array of 4 floats per group.
    unsafe {
        let mut acc = [vdupq_n_f32(-0.0); G];
        let mut i = 0;
        while i < n4 {
            let x = vld1q_f32(xr.as_ptr().add(i));
            for g in 0..G {
                let p = w.as_ptr().add((c0 + 4 * g) * n + i);
                let (r0, r1, r2, r3) = (
                    vld1q_f32(p),
                    vld1q_f32(p.add(n)),
                    vld1q_f32(p.add(2 * n)),
                    vld1q_f32(p.add(3 * n)),
                );
                // The 4×4 transpose: `c[k]` holds input `i + k` of the group's 4 rows.
                let (t0, t1) = (vtrn1q_f32(r0, r1), vtrn2q_f32(r0, r1));
                let (t2, t3) = (vtrn1q_f32(r2, r3), vtrn2q_f32(r2, r3));
                let lo = |a: float32x4_t, b: float32x4_t| {
                    vreinterpretq_f32_f64(vtrn1q_f64(
                        vreinterpretq_f64_f32(a),
                        vreinterpretq_f64_f32(b),
                    ))
                };
                let hi = |a: float32x4_t, b: float32x4_t| {
                    vreinterpretq_f32_f64(vtrn2q_f64(
                        vreinterpretq_f64_f32(a),
                        vreinterpretq_f64_f32(b),
                    ))
                };
                let c = [lo(t0, t2), lo(t1, t3), hi(t0, t2), hi(t1, t3)];
                acc[g] = vaddq_f32(acc[g], vmulq_f32(vdupq_laneq_f32::<0>(x), c[0]));
                acc[g] = vaddq_f32(acc[g], vmulq_f32(vdupq_laneq_f32::<1>(x), c[1]));
                acc[g] = vaddq_f32(acc[g], vmulq_f32(vdupq_laneq_f32::<2>(x), c[2]));
                acc[g] = vaddq_f32(acc[g], vmulq_f32(vdupq_laneq_f32::<3>(x), c[3]));
            }
            i += 4;
        }
        for g in 0..G {
            vst1q_f32(lanes[g].as_mut_ptr(), acc[g]);
        }
    }
    finish(lanes, xr, w, n, c0, n4, out);
}

// Index loops over the fixed-size register arrays: they unroll fully and pair groups with lanes.
#[allow(clippy::needless_range_loop)]
//
// The AVX2 path's group, in 128-bit vectors: a 4×4 transpose is four unpacks and four moves, and
// the lanes round exactly as the 256-bit ones do (mul then add, not FMA). The intrinsics are SSE,
// part of the x86_64 baseline, so this is an ordinary function that inlines into
// [`linear_rows_avx2`], which is compiled for AVX2 and holds the row loop: one call per
// projection rather than one per group.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn group_sse<const G: usize>(xr: &[f32], w: &[f32], n: usize, c0: usize, out: &mut [f32]) {
    use core::arch::x86_64::*;
    assert!(xr.len() >= n && w.len() >= (c0 + 4 * G) * n);
    let n4 = n - n % 4;
    let mut lanes = [[0.0f32; 4]; G];
    // SAFETY: SSE is part of the x86_64 baseline. Weight row `c0 + 4g + j` (`< c0 + 4G`) is read
    // as 4 unaligned floats at input `i < n4 ≤ n`, within `w` (asserted `(c0 + 4G) · n` long);
    // `xr[i..i + 4]` is within `xr` (asserted `n` long); the stores write a local 4-float group.
    unsafe {
        let mut acc = [_mm_set1_ps(-0.0); G];
        let mut i = 0;
        while i < n4 {
            let x = _mm_loadu_ps(xr.as_ptr().add(i));
            let xs = [
                _mm_shuffle_ps::<0x00>(x, x),
                _mm_shuffle_ps::<0x55>(x, x),
                _mm_shuffle_ps::<0xAA>(x, x),
                _mm_shuffle_ps::<0xFF>(x, x),
            ];
            for g in 0..G {
                let p = w.as_ptr().add((c0 + 4 * g) * n + i);
                let (r0, r1, r2, r3) = (
                    _mm_loadu_ps(p),
                    _mm_loadu_ps(p.add(n)),
                    _mm_loadu_ps(p.add(2 * n)),
                    _mm_loadu_ps(p.add(3 * n)),
                );
                let (t0, t1) = (_mm_unpacklo_ps(r0, r1), _mm_unpacklo_ps(r2, r3));
                let (t2, t3) = (_mm_unpackhi_ps(r0, r1), _mm_unpackhi_ps(r2, r3));
                let c = [
                    _mm_movelh_ps(t0, t1),
                    _mm_movehl_ps(t1, t0),
                    _mm_movelh_ps(t2, t3),
                    _mm_movehl_ps(t3, t2),
                ];
                for k in 0..4 {
                    acc[g] = _mm_add_ps(acc[g], _mm_mul_ps(xs[k], c[k]));
                }
            }
            i += 4;
        }
        for g in 0..G {
            _mm_storeu_ps(lanes[g].as_mut_ptr(), acc[g]);
        }
    }
    finish(lanes, xr, w, n, c0, n4, out);
}

/// [`linear_rows_into`] on the AVX2 path: the whole row loop compiled for AVX2, with
/// [`group_sse`] inlined into it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn linear_rows_avx2(x: &[f32], w: &[f32], n: usize, out: &mut [f32]) {
    rows_into(x, w, n, out, |xr, c, o| group_sse::<GROUPS>(xr, w, n, c, o));
}

// Index loops over the fixed-size register arrays: they unroll fully and pair groups with lanes.
#[allow(clippy::needless_range_loop)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn group_simd128<const G: usize>(xr: &[f32], w: &[f32], n: usize, c0: usize, out: &mut [f32]) {
    use core::arch::wasm32::*;
    assert!(xr.len() >= n && w.len() >= (c0 + 4 * G) * n);
    let n4 = n - n % 4;
    let mut lanes = [[0.0f32; 4]; G];
    // SAFETY: simd128 is enabled at compile time (the cfg). Weight row `c0 + 4g + j`
    // (`< c0 + 4G`) is read as 4 floats at input `i < n4 ≤ n`, within `w` (asserted
    // `(c0 + 4G) · n` long), and `xr[i..i + 4]` within `xr` (asserted `n` long); `v128_load`
    // has no alignment requirement; the stores write a local 4-float group.
    unsafe {
        let mut acc = [f32x4_splat(-0.0); G];
        let mut i = 0;
        while i < n4 {
            let x = v128_load(xr.as_ptr().add(i) as *const v128);
            let xs = [
                i32x4_shuffle::<0, 0, 0, 0>(x, x),
                i32x4_shuffle::<1, 1, 1, 1>(x, x),
                i32x4_shuffle::<2, 2, 2, 2>(x, x),
                i32x4_shuffle::<3, 3, 3, 3>(x, x),
            ];
            for g in 0..G {
                let p = w.as_ptr().add((c0 + 4 * g) * n + i);
                let (r0, r1, r2, r3) = (
                    v128_load(p as *const v128),
                    v128_load(p.add(n) as *const v128),
                    v128_load(p.add(2 * n) as *const v128),
                    v128_load(p.add(3 * n) as *const v128),
                );
                let t0 = i32x4_shuffle::<0, 4, 1, 5>(r0, r1);
                let t1 = i32x4_shuffle::<0, 4, 1, 5>(r2, r3);
                let t2 = i32x4_shuffle::<2, 6, 3, 7>(r0, r1);
                let t3 = i32x4_shuffle::<2, 6, 3, 7>(r2, r3);
                let c = [
                    i32x4_shuffle::<0, 1, 4, 5>(t0, t1),
                    i32x4_shuffle::<2, 3, 6, 7>(t0, t1),
                    i32x4_shuffle::<0, 1, 4, 5>(t2, t3),
                    i32x4_shuffle::<2, 3, 6, 7>(t2, t3),
                ];
                for k in 0..4 {
                    acc[g] = f32x4_add(acc[g], f32x4_mul(xs[k], c[k]));
                }
            }
            i += 4;
        }
        for g in 0..G {
            v128_store(lanes[g].as_mut_ptr() as *mut v128, acc[g]);
        }
    }
    finish(lanes, xr, w, n, c0, n4, out);
}

/// The row loop: for each row of `x`, [`GROUPS`] of 4 outputs at a time by `group(row, first
/// output, row's outputs)`, then the last few by the scalar sum. `n > 0`, at least one output.
#[inline(always)]
fn rows_into(
    x: &[f32],
    w: &[f32],
    n: usize,
    out: &mut [f32],
    mut group: impl FnMut(&[f32], usize, &mut [f32]),
) {
    let outputs = w.len() / n;
    for (xr, o) in x.chunks_exact(n).zip(out.chunks_exact_mut(outputs)) {
        let mut c = 0;
        while c + 4 * GROUPS <= outputs {
            group(xr, c, o);
            c += 4 * GROUPS;
        }
        for c in c..outputs {
            let mut a = -0.0f32;
            for (&xi, &wi) in xr.iter().zip(&w[c * n..(c + 1) * n]) {
                a += xi * wi;
            }
            o[c] = a;
        }
    }
}

/// `x [rows, n] · w[outputs, n]ᵀ` into `out [rows, outputs]` on `path`, one row at a time:
/// [`GROUPS`] of 4 outputs together, then the last few by the scalar sum. `n > 0`.
pub(super) fn linear_rows_into(path: Path, x: &[f32], w: &[f32], n: usize, out: &mut [f32]) {
    if w.len() < n {
        // No outputs: nothing to write (and no row of outputs to split `out` into).
        return;
    }
    match path {
        Path::Scalar => rows_into(x, w, n, out, |xr, c, o| {
            group_scalar::<GROUPS>(xr, w, n, c, o)
        }),
        #[cfg(target_arch = "aarch64")]
        Path::Neon => rows_into(x, w, n, out, |xr, c, o| {
            group_neon::<GROUPS>(xr, w, n, c, o)
        }),
        // SAFETY: `Path::Avx2` carries an `Avx2Detected`, which only `best_path` builds, after
        // `is_x86_feature_detected!("avx2")`.
        #[cfg(target_arch = "x86_64")]
        Path::Avx2(_) => {
            debug_assert!(is_x86_feature_detected!("avx2"));
            unsafe { linear_rows_avx2(x, w, n, out) }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Path::Simd128 => rows_into(x, w, n, out, |xr, c, o| {
            group_simd128::<GROUPS>(xr, w, n, c, o)
        }),
    }
}
