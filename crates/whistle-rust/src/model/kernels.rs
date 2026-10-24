//! The dense f32 projection kernel, `x · wᵀ`, and the library's worker-thread setting
//! ([`set_threads`]).
//!
//! **The order.** Every output is the plain sequential sum `((-0 + x0·w0) + x1·w1) + …` in input
//! order, each product rounded then added (no FMA): exactly what the reference port's
//! `iter().map(|(a, b)| a * b).sum()` computes, so every stage bound and the token contract are
//! unchanged by speed. It vectorizes *across outputs* instead of within a sum: a panel of
//! [`PANEL`] weight rows is transposed once into `[n, PANEL]` scratch, then each input row
//! broadcasts `x[i]` against row `i` of the panel, one SIMD lane per output. The portable scalar
//! path, NEON (aarch64), AVX2 (x86_64, detected at run time) and wasm `simd128` therefore agree
//! bit for bit, and threads only choose which outputs a worker owns, never how one is summed,
//! so results do not depend on the thread count either.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Outputs computed together: one lane each.
const PANEL: usize = 16;

/// Input rows sharing one pass over a transposed panel.
const ROWS: usize = 4;

/// The most worker threads any setting gives: a fixed cap (not the machine's core count, so a
/// setting means the same everywhere), well past where these projections stop scaling.
pub const MAX_THREADS: usize = 16;

/// 0 means "not set": [`threads`] then uses `available_parallelism().min(4)`.
static THREADS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// A per-calling-thread override of [`THREADS`] for the duration of [`with_threads`].
    static OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Sets how many worker threads a transcription's large projections may split across (1 runs
/// everything on the caller's thread; 0 restores the default, `available_parallelism().min(4)`;
/// anything above [`MAX_THREADS`] is [`MAX_THREADS`]). Process-wide. Results are bit-identical
/// for every setting. wasm32 is always single-threaded.
pub fn set_threads(n: usize) {
    THREADS.store(n.min(MAX_THREADS), Ordering::Relaxed);
}

/// The worker-thread count [`set_threads`] (or an enclosing [`with_threads`]) chose, or the
/// default; always in `1..=MAX_THREADS`.
pub fn threads() -> usize {
    if cfg!(target_arch = "wasm32") {
        return 1;
    }
    let n = OVERRIDE
        .with(Cell::get)
        .unwrap_or_else(|| THREADS.load(Ordering::Relaxed));
    match n {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get().min(4)),
        n => n.min(MAX_THREADS),
    }
}

/// Runs `f` with `n` worker threads for the projections it starts on this thread (0: the
/// default), whatever the process-wide setting, so tests can compare thread counts without
/// racing each other on [`set_threads`]. Not part of the stable API.
#[doc(hidden)]
pub fn with_threads<R>(n: usize, f: impl FnOnce() -> R) -> R {
    let previous = OVERRIDE.with(|o| o.replace(Some(n.min(MAX_THREADS))));
    struct Restore(Option<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OVERRIDE.with(|o| o.set(self.0));
        }
    }
    let _restore = Restore(previous);
    f()
}

/// Below this many multiply-adds a projection stays on one thread: spawning costs more.
/// Measured 2026-10-24 (Apple M5, release, loaded machine; `parallel_threshold_timing`, 64 rows
/// of 256, best of 30): 2¹⁸ MACs 35 µs on 1 thread vs 44 µs on 4, 2¹⁹ 52 vs 54 µs, 2²⁰ 86 vs
/// 58 µs, 2²¹ 156 vs 80 µs; the crossover is between 2¹⁹ and 2²⁰.
const PARALLEL_MIN_MACS: usize = 1 << 20;

/// Fewer input rows than this per worker stay on one thread. A decoder step's single-row
/// projections (the vocabulary head included) measured slower split across spawned threads
/// (5.03 ms per token on 1 thread, 6.01 ms on 4) than on one, so until a persistent pool
/// exists they do not split at all.
const MIN_ROWS_PER_WORKER: usize = 16;

/// Which implementation of the order to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Path {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// Only [`best_path`] can build the token, after detecting AVX2.
    #[cfg(target_arch = "x86_64")]
    Avx2(Avx2Detected),
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Simd128,
}

/// Proof that AVX2 was detected at run time: its field is private to this module and only
/// [`best_path`] constructs it, after `is_x86_feature_detected!("avx2")`.
#[cfg(target_arch = "x86_64")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Avx2Detected(());

/// The fastest path this machine supports.
pub(crate) fn best_path() -> Path {
    #[cfg(target_arch = "aarch64")]
    return Path::Neon;
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return Path::Avx2(Avx2Detected(()));
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return Path::Simd128;
    #[allow(unreachable_code)]
    Path::Scalar
}

/// `R` input rows (each `n` long) against a transposed panel `wt [n, PANEL]`:
/// `out[r][c] = Σᵢ x[r][i] · wt[i][c]`, in order from `-0.0`.
#[inline(always)]
fn panel_scalar<const R: usize>(x: [&[f32]; R], wt: &[f32], n: usize) -> [[f32; PANEL]; R] {
    let mut acc = [[-0.0f32; PANEL]; R];
    for (i, row) in wt.chunks_exact(PANEL).take(n).enumerate() {
        for r in 0..R {
            let xi = x[r][i];
            for (a, &w) in acc[r].iter_mut().zip(row) {
                *a += xi * w;
            }
        }
    }
    acc
}

// Index loops over the fixed-size register arrays: they unroll fully and pair rows with lanes.
#[allow(clippy::needless_range_loop)]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn panel_neon<const R: usize>(x: [&[f32]; R], wt: &[f32], n: usize) -> [[f32; PANEL]; R] {
    use core::arch::aarch64::*;
    assert!(wt.len() >= n * PANEL && x.iter().all(|r| r.len() >= n));
    // SAFETY: NEON is part of the aarch64 baseline. Panel row `i < n` is read as four 4-float
    // loads at `i · PANEL + 4k`, all within `wt` (asserted `n · PANEL` long); `x[r][i]` is in
    // bounds by the same assert; the stores write a local array of `PANEL` floats per row.
    unsafe {
        let mut acc = [[vdupq_n_f32(-0.0); 4]; R];
        for i in 0..n {
            let p = wt.as_ptr().add(i * PANEL);
            let w = [
                vld1q_f32(p),
                vld1q_f32(p.add(4)),
                vld1q_f32(p.add(8)),
                vld1q_f32(p.add(12)),
            ];
            for r in 0..R {
                let xi = vdupq_n_f32(*x[r].get_unchecked(i));
                for k in 0..4 {
                    acc[r][k] = vaddq_f32(acc[r][k], vmulq_f32(xi, w[k]));
                }
            }
        }
        let mut out = [[0.0f32; PANEL]; R];
        for r in 0..R {
            for k in 0..4 {
                vst1q_f32(out[r].as_mut_ptr().add(4 * k), acc[r][k]);
            }
        }
        out
    }
}

// Index loops over the fixed-size register arrays: they unroll fully and pair rows with lanes.
#[allow(clippy::needless_range_loop)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn panel_avx2<const R: usize>(x: [&[f32]; R], wt: &[f32], n: usize) -> [[f32; PANEL]; R] {
    use core::arch::x86_64::*;
    assert!(wt.len() >= n * PANEL && x.iter().all(|r| r.len() >= n));
    let mut acc = [[_mm256_set1_ps(-0.0); 2]; R];
    for i in 0..n {
        // SAFETY: the caller checked AVX2 is available. Panel row `i < n` is two unaligned
        // 8-float loads within `wt` (asserted `n · PANEL` long); `x[r][i]` is in bounds by the
        // same assert.
        let p = wt.as_ptr().add(i * PANEL);
        let w = [_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8))];
        for r in 0..R {
            let xi = _mm256_set1_ps(*x[r].get_unchecked(i));
            for k in 0..2 {
                acc[r][k] = _mm256_add_ps(acc[r][k], _mm256_mul_ps(xi, w[k]));
            }
        }
    }
    let mut out = [[0.0f32; PANEL]; R];
    for r in 0..R {
        for k in 0..2 {
            // SAFETY: writes 8 floats at `8k`, within the local `PANEL`-float row.
            _mm256_storeu_ps(out[r].as_mut_ptr().add(8 * k), acc[r][k]);
        }
    }
    out
}

// Index loops over the fixed-size register arrays: they unroll fully and pair rows with lanes.
#[allow(clippy::needless_range_loop)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn panel_simd128<const R: usize>(x: [&[f32]; R], wt: &[f32], n: usize) -> [[f32; PANEL]; R] {
    use core::arch::wasm32::*;
    assert!(wt.len() >= n * PANEL && x.iter().all(|r| r.len() >= n));
    // SAFETY: simd128 is enabled at compile time (the cfg). Panel row `i < n` is four 4-float
    // loads within `wt` (asserted `n · PANEL` long), `v128_load` has no alignment requirement;
    // `x[r][i]` is in bounds by the same assert; the stores write a local `PANEL`-float row.
    unsafe {
        let mut acc = [[f32x4_splat(-0.0); 4]; R];
        for i in 0..n {
            let p = wt.as_ptr().add(i * PANEL) as *const v128;
            let w = [
                v128_load(p),
                v128_load(p.add(1)),
                v128_load(p.add(2)),
                v128_load(p.add(3)),
            ];
            for r in 0..R {
                let xi = f32x4_splat(*x[r].get_unchecked(i));
                for k in 0..4 {
                    acc[r][k] = f32x4_add(acc[r][k], f32x4_mul(xi, w[k]));
                }
            }
        }
        let mut out = [[0.0f32; PANEL]; R];
        for r in 0..R {
            for k in 0..4 {
                v128_store(out[r].as_mut_ptr().add(4 * k) as *mut v128, acc[r][k]);
            }
        }
        out
    }
}

/// [`panel_scalar`] on `path`'s implementation.
#[inline(always)]
fn panel<const R: usize>(path: Path, x: [&[f32]; R], wt: &[f32], n: usize) -> [[f32; PANEL]; R] {
    match path {
        Path::Scalar => panel_scalar(x, wt, n),
        #[cfg(target_arch = "aarch64")]
        Path::Neon => panel_neon(x, wt, n),
        // SAFETY: `Path::Avx2` carries an `Avx2Detected`, which only `best_path` builds, after
        // `is_x86_feature_detected!("avx2")`.
        #[cfg(target_arch = "x86_64")]
        Path::Avx2(_) => {
            debug_assert!(is_x86_feature_detected!("avx2"));
            unsafe { panel_avx2(x, wt, n) }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Path::Simd128 => panel_simd128(x, wt, n),
    }
}

/// `x [rows, n] · w[outputs, n]ᵀ` into `out [rows, outputs]` on `path`, one thread.
pub(crate) fn linear_into(path: Path, x: &[f32], w: &[f32], n: usize, out: &mut [f32]) {
    if n == 0 {
        out.fill(0.0);
        return;
    }
    let (rows, outputs) = (x.len() / n, w.len() / n);
    let xr = |r: usize| &x[r * n..(r + 1) * n];
    let mut wt = vec![0.0f32; n * PANEL];
    for c0 in (0..outputs).step_by(PANEL) {
        let width = PANEL.min(outputs - c0);
        // Transpose this panel's weight rows; lanes past `outputs` stay zero and are dropped.
        for (c, wrow) in w[c0 * n..(c0 + width) * n].chunks_exact(n).enumerate() {
            for (i, &v) in wrow.iter().enumerate() {
                wt[i * PANEL + c] = v;
            }
        }
        if width < PANEL {
            for i in 0..n {
                wt[i * PANEL + width..(i + 1) * PANEL].fill(0.0);
            }
        }
        let mut store = |r: usize, o: &[f32; PANEL]| {
            out[r * outputs + c0..r * outputs + c0 + width].copy_from_slice(&o[..width]);
        };
        let mut r = 0;
        while r + ROWS <= rows {
            let o = panel::<ROWS>(path, [xr(r), xr(r + 1), xr(r + 2), xr(r + 3)], &wt, n);
            for (k, o) in o.iter().enumerate() {
                store(r + k, o);
            }
            r += ROWS;
        }
        while r < rows {
            store(r, &panel::<1>(path, [xr(r)], &wt, n)[0]);
            r += 1;
        }
    }
}

/// `x [T, inputs] @ w[outputs, inputs]ᵀ -> [T, outputs]` on `path` with up to `threads`
/// workers (clamped to `1..=MAX_THREADS`), each owning a contiguous band of input rows; small
/// projections and those with few rows stay on the caller's thread.
pub(crate) fn linear_with(
    path: Path,
    threads: usize,
    x: &[f32],
    w: &[f32],
    inputs: usize,
    outputs: usize,
) -> Vec<f32> {
    let rows = x.len().checked_div(inputs).unwrap_or(0);
    let mut out = vec![0.0f32; rows * outputs];
    let macs = rows.saturating_mul(inputs).saturating_mul(outputs);
    let workers = if macs < PARALLEL_MIN_MACS {
        1
    } else {
        threads
            .clamp(1, MAX_THREADS)
            .min(rows / MIN_ROWS_PER_WORKER)
    };
    if workers <= 1 || inputs == 0 {
        linear_into(path, x, w, inputs, &mut out);
        return out;
    }
    let per = rows.div_ceil(workers);
    std::thread::scope(|s| {
        for (xc, oc) in x.chunks(per * inputs).zip(out.chunks_mut(per * outputs)) {
            s.spawn(move || linear_into(path, xc, w, inputs, oc));
        }
    });
    out
}

#[cfg(test)]
mod tests;
