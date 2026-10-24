//! The dense order of a packed matrix (`kernels::packed`, `Order::Dense`): dequantize a band of
//! rows, then multiply it with the dense kernel straight away, while it is in cache. The
//! encoder, the stem and the decoder's cross-attention keys and values project this way.
//!
//! **The decode** gives exactly `Tensor::to_f32`'s bits. Each element goes through the same
//! operations in the same order: the codebook value, times the group's norm, times the f32
//! `1/sqrt(group)`, then the Walsh–Hadamard butterflies level by level, `(x + y, x − y)` with
//! `x` the lower index. A level's butterflies are independent of each other, so computing them
//! four at a time changes no bits. NEON folds the norm and scale into a per-group codebook
//! table (`(cb[c] · norm) · scale`, the same two multiplies per element), decodes 16 indices per step with the byte-table lookups
//! `packed` uses, then does levels 1, 2, 4 and 8 in registers (unzip/zip and 64-bit transposes
//! for 1 and 2). Each higher level is a vector add and subtract over the group. Other targets run
//! the scalar decode, which is `quant`'s.
//!
//! **Threads** split the outputs: each worker owns a contiguous slice of the band's rows,
//! decodes them in bands of [`BAND`] and multiplies each band by every input row. Every output is
//! still one sequential sum (the dense kernel's), so the result does not depend on the thread
//! count, and no full dequantized copy is ever made.

use super::super::{
    linear_band_into, linear_into, Path, MAX_THREADS, MIN_ROWS_PER_WORKER, PARALLEL_MIN_MACS, ROWS,
};
use super::{code, PackedMatrix};
use crate::quant::walsh_hadamard;

/// Rows decoded at once: 64 rows of 512 are 128 KB, so a band stays in cache while every
/// input row multiplies it.
const BAND: usize = 64;

impl PackedMatrix {
    /// Rows `first..first + out.len() / cols` dequantized into `out` (row-major, padding
    /// dropped), exactly as `Tensor::to_f32`'s rows. `padded` is `in_pad` long, and is only used
    /// when rows are padded. The caller keeps the rows within the matrix.
    pub(crate) fn band_into(&self, path: Path, first: usize, out: &mut [f32], padded: &mut [f32]) {
        match (path, self.bits) {
            #[cfg(target_arch = "aarch64")]
            (Path::Neon, 2) => self.band_loop(first, out, padded, |row, g, norm, chunk| {
                self.decode_group_neon::<2>(row, g, norm, chunk)
            }),
            #[cfg(target_arch = "aarch64")]
            (Path::Neon, _) => self.band_loop(first, out, padded, |row, g, norm, chunk| {
                self.decode_group_neon::<4>(row, g, norm, chunk)
            }),
            // SAFETY: `Path::Avx2` carries an `Avx2Detected`, which only `best_path` builds,
            // after `is_x86_feature_detected!("avx2")`.
            #[cfg(target_arch = "x86_64")]
            (Path::Avx2(_), _) => {
                debug_assert!(is_x86_feature_detected!("avx2"));
                unsafe { self.band_into_avx2(first, out, padded) }
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            (Path::Simd128, 2) => self.band_loop(first, out, padded, |row, g, norm, chunk| {
                self.decode_group_simd128::<2>(row, g, norm, chunk)
            }),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            (Path::Simd128, _) => self.band_loop(first, out, padded, |row, g, norm, chunk| {
                self.decode_group_simd128::<4>(row, g, norm, chunk)
            }),
            _ => self.band_loop(first, out, padded, |row, g, norm, chunk| {
                self.decode_group_scalar(row, g, norm, chunk)
            }),
        }
    }

    /// The loop of [`Self::band_into`] over rows and groups, `group(row bytes, group, norm,
    /// values)` decoding one group in place. Inlined into every path, so a SIMD path compiled
    /// for its feature holds the whole band.
    #[inline(always)]
    pub(super) fn band_loop(
        &self,
        first: usize,
        out: &mut [f32],
        padded: &mut [f32],
        group: impl Fn(&[u8], usize, f32, &mut [f32]),
    ) {
        let cols = self.cols();
        let groups = self.in_pad / self.group;
        // No closure in here: one defined in this (featureless) function could not inline a
        // SIMD path's `group`, which is compiled for its feature, so it would be one call per
        // group.
        let packed = self.cq.parts().packed;
        for (i, row_out) in out.chunks_exact_mut(cols).enumerate() {
            let r = first + i;
            let row = &packed[r * self.row_bytes..(r + 1) * self.row_bytes];
            let whole = cols == self.in_pad;
            let dst: &mut [f32] = if whole { &mut *row_out } else { &mut *padded };
            for (g, chunk) in dst.chunks_exact_mut(self.group).enumerate() {
                group(row, g, self.norms[r * groups + g], chunk);
            }
            if !whole {
                row_out.copy_from_slice(&padded[..cols]);
            }
        }
    }

    /// The scalar decode of group `g`: `quant`'s operations exactly.
    fn decode_group_scalar(&self, row: &[u8], g: usize, norm: f32, chunk: &mut [f32]) {
        let cb = self.cq.parts().codebook;
        for (k, v) in chunk.iter_mut().enumerate() {
            *v = cb[code(row, g * self.group + k, self.bits)];
        }
        for v in chunk.iter_mut() {
            *v *= norm;
        }
        walsh_hadamard(chunk);
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn decode_group_neon<const B: u32>(&self, row: &[u8], g: usize, norm: f32, chunk: &mut [f32]) {
        use core::arch::aarch64::*;
        let n = self.group;
        // The butterflies need a power-of-two group (`PackedMatrix::new` admits no other).
        assert!(chunk.len() == n && n.is_power_of_two() && n >= 16);
        assert!(row.len() * 8 >= (g + 1) * n * B as usize);
        assert!(self.cq.parts().codebook.len() == 1 << B);
        // The same scale `walsh_hadamard` multiplies by.
        let scale = (1.0 / (n as f64).sqrt()) as f32;
        let tables = super::DecodeTables::new::<B>();
        // SAFETY: NEON is part of the aarch64 baseline. The codebook loads read entries
        // `4i..4i + 4` with `4i + 4 ≤ 2^B`, its asserted length. The index loads read 4 (`B` = 2)
        // or 8 bytes at `(g · n + k0) · B / 8`, `k0 + 16 ≤ n`, within `row` (asserted at least
        // `(g + 1) · n · B / 8` bytes), unaligned via `read_unaligned`. The float loads and
        // stores touch `chunk[i..i + 4]` with `i + 4 ≤ n` (asserted `n` long): the group `n` is
        // a power of two of at least 16 (asserted), so `16 | n` for the register levels and, at
        // every memory level `half ≥ 16`, `2 · half` divides `n` and each partner
        // `i + half < block + 2 · half ≤ n`. The table lookups index registers only.
        unsafe {
            // The group's lookup table holds `(cb[c] · norm) · scale` per code: `quant`'s two
            // multiplies, done once per code instead of once per element (the same bits).
            let (vn, vs) = (vdupq_n_f32(norm), vdupq_n_f32(scale));
            let cb = self.cq.parts().codebook.as_ptr();
            let entry = |i: usize| {
                vreinterpretq_u8_f32(vmulq_f32(vmulq_f32(vld1q_f32(cb.add(4 * i)), vn), vs))
            };
            let lut = if B == 2 {
                uint8x16x4_t(entry(0), entry(0), entry(0), entry(0))
            } else {
                uint8x16x4_t(entry(0), entry(1), entry(2), entry(3))
            };
            let lo64 = |a: float32x4_t, b: float32x4_t| {
                vreinterpretq_f32_f64(vtrn1q_f64(
                    vreinterpretq_f64_f32(a),
                    vreinterpretq_f64_f32(b),
                ))
            };
            let hi64 = |a: float32x4_t, b: float32x4_t| {
                vreinterpretq_f32_f64(vtrn2q_f64(
                    vreinterpretq_f64_f32(a),
                    vreinterpretq_f64_f32(b),
                ))
            };
            let p = chunk.as_mut_ptr();
            // Levels 1 to 8 of the 16 values at `k0`, in registers.
            let low_levels = |k0: usize| {
                let src = row.as_ptr().add((g * n + k0) * B as usize / 8);
                let w = tables.decode16::<B>(src, &lut);
                // Level 1 (pairs within a vector).
                let half1 = |a: float32x4_t, b: float32x4_t| {
                    let (t1, t2) = (vuzp1q_f32(a, b), vuzp2q_f32(a, b));
                    let (s, d) = (vaddq_f32(t1, t2), vsubq_f32(t1, t2));
                    (vzip1q_f32(s, d), vzip2q_f32(s, d))
                };
                let (a0, a1) = half1(w[0], w[1]);
                let (a2, a3) = half1(w[2], w[3]);
                // Level 2 (halves of a vector).
                let half2 = |a: float32x4_t, b: float32x4_t| {
                    let (t1, t2) = (lo64(a, b), hi64(a, b));
                    let (s, d) = (vaddq_f32(t1, t2), vsubq_f32(t1, t2));
                    (lo64(s, d), hi64(s, d))
                };
                let (b0, b1) = half2(a0, a1);
                let (b2, b3) = half2(a2, a3);
                // Level 4 (neighbouring vectors), then level 8 (vectors two apart).
                let (c0, c1) = (vaddq_f32(b0, b1), vsubq_f32(b0, b1));
                let (c2, c3) = (vaddq_f32(b2, b3), vsubq_f32(b2, b3));
                [
                    vaddq_f32(c0, c2),
                    vaddq_f32(c1, c3),
                    vsubq_f32(c0, c2),
                    vsubq_f32(c1, c3),
                ]
            };
            // Levels 16 and above go through memory. (Keeping 16 and 32 in registers too, 64
            // values as 16 vectors, measured no faster on a loaded machine.)
            for k0 in (0..n).step_by(16) {
                for (i, x) in low_levels(k0).iter().enumerate() {
                    vst1q_f32(p.add(k0 + 4 * i), *x);
                }
            }
            let mut half = 16;
            while half < n {
                for block in (0..n).step_by(2 * half) {
                    for i in (block..block + half).step_by(4) {
                        let (x, y) = (vld1q_f32(p.add(i)), vld1q_f32(p.add(i + half)));
                        vst1q_f32(p.add(i), vaddq_f32(x, y));
                        vst1q_f32(p.add(i + half), vsubq_f32(x, y));
                    }
                }
                half *= 2;
            }
        }
    }

    /// The dense order's projection: `x [T, cols] · w[first..first + count]ᵀ -> [T, count]`.
    /// Workers own contiguous slices of the outputs (module header).
    pub(super) fn project_dense(
        &self,
        path: Path,
        threads: usize,
        x: &[f32],
        first: usize,
        count: usize,
    ) -> Vec<f32> {
        let cols = self.cols();
        let rows = x.len().checked_div(cols).unwrap_or(0);
        let mut out = vec![0.0f32; rows * count];
        if rows == 0 || count == 0 {
            return out;
        }
        let macs = rows.saturating_mul(cols).saturating_mul(count);
        let workers = if macs < PARALLEL_MIN_MACS {
            1
        } else {
            threads
                .clamp(1, MAX_THREADS)
                .min(rows / MIN_ROWS_PER_WORKER)
                .min(count.div_ceil(BAND))
        };
        if workers <= 1 {
            self.slice_into(path, x, first, count, &mut out);
            return out;
        }
        // Slices of whole bands, so every worker but the last decodes full bands.
        let per = count.div_ceil(workers).next_multiple_of(BAND);
        let mut parts: Vec<(usize, usize, Vec<f32>)> = (0..count)
            .step_by(per)
            .map(|c0| {
                let width = per.min(count - c0);
                (c0, width, vec![0.0f32; rows * width])
            })
            .collect();
        std::thread::scope(|s| {
            for (c0, width, o) in parts.iter_mut() {
                let (c0, width) = (*c0, *width);
                s.spawn(move || self.slice_into(path, x, first + c0, width, o));
            }
        });
        for (c0, width, o) in &parts {
            for (dst, src) in out.chunks_exact_mut(count).zip(o.chunks_exact(*width)) {
                dst[*c0..*c0 + *width].copy_from_slice(src);
            }
        }
        out
    }

    /// Outputs `first..first + count` for every row of `x` into `out [T, count]` on this thread,
    /// band by band: decode, then multiply while the band is in cache.
    fn slice_into(&self, path: Path, x: &[f32], first: usize, count: usize, out: &mut [f32]) {
        let cols = self.cols();
        let rows = x.len() / cols;
        let mut band = vec![0.0f32; BAND.min(count) * cols];
        let mut padded = vec![0.0f32; self.in_pad];
        let mut part = vec![
            0.0f32;
            if rows < ROWS {
                rows * BAND.min(count)
            } else {
                0
            }
        ];
        for b0 in (0..count).step_by(BAND) {
            let width = BAND.min(count - b0);
            let w = &mut band[..width * cols];
            self.band_into(path, first + b0, w, &mut padded);
            if rows >= ROWS {
                // Straight into place: no copy of the band's outputs.
                linear_band_into(path, x, w, cols, out, count, b0);
                continue;
            }
            let o = &mut part[..rows * width];
            linear_into(path, x, w, cols, o);
            for (dst, src) in out.chunks_exact_mut(count).zip(o.chunks_exact(width)) {
                dst[b0..b0 + width].copy_from_slice(src);
            }
        }
    }
}
