//! The packed-weight projection (`kernels`' CQ counterpart of the dense `linear_into`): `x · wᵀ`
//! straight from a CQ record's 2- or 4-bit indices and per-group norms, with no f32 weights.
//!
//! **The math.** A CQ row's group dequantizes to `w_g = (cb[idx] · norm_g) @ H`, `H` the
//! symmetric orthonormal Walsh–Hadamard matrix, so `w_g · x_g = norm_g · (cb[idx] · (H x_g))`.
//! Each input row is padded to the record's padded width and rotated once per group by
//! `quant::walsh_hadamard`; each output is then `Σ_g norm_g · Σ_k cb[idx_{g,k}] · xr_{g,k}`.
//!
//! **The order** (every path, bit for bit): within a group, 16 lanes, lane `j` summing inputs
//! `k ≡ j (mod 16)` in order from `-0.0`, product rounded then added (no FMA); the lanes reduce
//! as `v_l = (a_l + a_{4+l}) + (a_{8+l} + a_{12+l})`, then `(v_0 + v_1) + (v_2 + v_3)`; the
//! output is the sequential sum from `-0.0` over groups of `norm_g · s_g`. This rounds
//! differently from dequantizing then summing densely, so it is an experiment the token
//! contract decides (`docs/design/performance.md`). NEON decodes 16 indices with byte shuffles
//! (`vqtbl1q_u8` / `vqtbl4q_u8` into the codebook's bytes), AVX2 with `_mm256_permutevar8x32_ps`
//! lookups into the codebook (`packed_x86`), simd128 with `i8x16_swizzle` into its byte planes
//! (`packed_wasm`); each keeps the scalar order's lanes and reduction, so all agree bit for bit.
//!
//! **The dense order.** A matrix bound with [`Order::Dense`] (the encoder's, the stem's and the
//! decoder's cross-attention keys and values, which project many frames, the keys and values
//! once per call over the encoder's, and have no cached step) instead dequantizes a band of rows
//! at a time and runs the dense kernel on it (`band`): the same bits as the fully dequantized
//! matrix, the decode amortized over every input row, and only the band resident. The rotated
//! order re-decodes the weights for every 4 input rows, which made the 30 s encoder 3–4×
//! slower.

use super::{Path, MAX_THREADS, MIN_ROWS_PER_WORKER, PARALLEL_MIN_MACS};
use crate::quant::{fp16_to_f32, walsh_hadamard, CqRows};

/// Inputs summed per lane step: 16 lanes, 4 vectors of 4.
const LANES: usize = 16;

/// Input rows sharing one decode of the weights.
const ROWS: usize = 4;

/// How a packed matrix projects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Order {
    /// From the rotated activations and the packed indices (module header): the decoder's
    /// matrices, so its full pass and its cached step share one kernel.
    Rotated,
    /// Band by band through the dense kernel, bit-identical to the dequantized matrix.
    Dense,
}

/// A CQ matrix kept packed for projection: its record, its norms as f32 (`[rows, groups]`) and
/// its codebook as little-endian bytes (the NEON lookup table).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PackedMatrix {
    cq: CqRows,
    order: Order,
    norms: Vec<f32>,
    lut: [u8; 64],
    /// The codebook as f32, zero-padded to 16 entries (the AVX2 and simd128 tables).
    cbf: [f32; 16],
    bits: u32,
    group: usize,
    in_pad: usize,
    row_bytes: usize,
    /// wasm simd128, rotated order only: every index as its own byte, `[rows, in_pad]`, so the
    /// kernel loads 16 codes with one `v128_load` instead of unpacking them (`packed_wasm`).
    /// wasm has no per-lane variable shift, and the unpack was most of the decoder's kernel;
    /// this costs one byte per weight (`docs/design/performance.md`). Native keeps none.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    codes: Vec<u8>,
}

/// Index `k` of an LSB-first stream of `bits`-bit fields, `bits` 2 or 4 (a field never
/// straddles a byte).
#[inline(always)]
fn code(row: &[u8], k: usize, bits: u32) -> usize {
    let bit = k * bits as usize;
    usize::from(row[bit / 8] >> (bit % 8)) & ((1 << bits) - 1)
}

impl PackedMatrix {
    /// `cq` as a packed matrix, or `None` when its width or group does not fit the kernel
    /// (only 2- and 4-bit records with full codebooks and power-of-two groups of at least 16
    /// do; the caller then keeps the matrix dense). The band decodes' butterflies rely on the
    /// power of two: every level's partner `i + half` stays inside the group.
    pub(crate) fn new(cq: CqRows, order: Order) -> Option<PackedMatrix> {
        let p = cq.parts();
        let fits = matches!(p.bits, 2 | 4)
            && p.codebook.len() == 1 << p.bits
            && p.group.is_power_of_two()
            && p.group >= LANES;
        if !fits {
            return None;
        }
        let mut lut = [0u8; 64];
        for (slot, v) in lut.chunks_exact_mut(4).zip(p.codebook) {
            slot.copy_from_slice(&v.to_le_bytes());
        }
        let mut cbf = [0.0f32; 16];
        cbf[..p.codebook.len()].copy_from_slice(p.codebook);
        let norms = p
            .norms
            .chunks_exact(2)
            .take(cq.rows() * (p.in_pad / p.group))
            .map(|b| fp16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect();
        let (bits, group, in_pad, row_bytes) = (p.bits, p.group, p.in_pad, p.row_bytes);
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        let codes = if order == Order::Rotated {
            p.packed
                .chunks_exact(row_bytes)
                .take(cq.rows())
                .flat_map(|row| (0..in_pad).map(move |k| code(row, k, bits) as u8))
                .collect()
        } else {
            Vec::new()
        };
        Some(PackedMatrix {
            cq,
            order,
            norms,
            lut,
            cbf,
            bits,
            group,
            in_pad,
            row_bytes,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            codes,
        })
    }

    pub(crate) fn rows(&self) -> usize {
        self.cq.rows()
    }

    pub(crate) fn cols(&self) -> usize {
        self.cq.cols()
    }

    /// The whole matrix dequantized, exactly as `Tensor::to_f32` (padding dropped). Cannot
    /// fail: `new` only accepts widths whose every code has a codebook entry.
    pub(crate) fn dense(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows() * self.cols()];
        let mut padded = vec![0.0f32; self.in_pad];
        self.band_into(super::best_path(), 0, &mut out, &mut padded);
        out
    }

    /// Row `r` dequantized into `out` (`cols` values), as [`Self::dense`]'s row `r`, padded
    /// records included. A row past the matrix or an `out` of another length is an error.
    pub(crate) fn row_into(&self, r: usize, out: &mut [f32]) -> Result<(), crate::Error> {
        if r >= self.rows() || out.len() != self.cols() {
            return Err(crate::Error::Input(format!(
                "row {r} of a [{}, {}] matrix into {} values",
                self.rows(),
                self.cols(),
                out.len()
            )));
        }
        // The padded scratch is only touched when rows are padded.
        let mut padded = if self.cols() == self.in_pad {
            Vec::new()
        } else {
            vec![0.0f32; self.in_pad]
        };
        self.band_into(super::best_path(), r, out, &mut padded);
        Ok(())
    }

    /// The order the matrix projects in.
    #[cfg(test)]
    pub(crate) fn order(&self) -> Order {
        self.order
    }

    /// Whether `other` holds the same record (indices, norms, codebook and layout): equal
    /// values without dequantizing either.
    pub(crate) fn same_record(&self, other: &PackedMatrix) -> bool {
        self.cq == other.cq
    }

    /// The input rows of `x` (`[T, cols]`) padded to `in_pad` and rotated group by group.
    fn rotate(&self, x: &[f32]) -> Vec<f32> {
        let cols = self.cols();
        let rows = x.len().checked_div(cols).unwrap_or(0);
        let mut xr = vec![0.0f32; rows * self.in_pad];
        for (src, dst) in x.chunks_exact(cols).zip(xr.chunks_exact_mut(self.in_pad)) {
            dst[..cols].copy_from_slice(src);
            for chunk in dst.chunks_exact_mut(self.group) {
                walsh_hadamard(chunk);
            }
        }
        xr
    }

    /// `x [T, cols] · w[first..first + count]ᵀ -> [T, count]` on `path` with up to `threads`
    /// workers, each owning a band of input rows (the dense kernel's split rule), so the
    /// result does not depend on the thread count.
    pub(crate) fn project(
        &self,
        path: Path,
        threads: usize,
        x: &[f32],
        first: usize,
        count: usize,
    ) -> Vec<f32> {
        if self.order == Order::Dense {
            return self.project_dense(path, threads, x, first, count);
        }
        let rows = x.len().checked_div(self.cols()).unwrap_or(0);
        let mut out = vec![0.0f32; rows * count];
        if count == 0 || rows == 0 {
            return out;
        }
        let xr = self.rotate(x);
        let macs = rows.saturating_mul(self.in_pad).saturating_mul(count);
        let workers = if macs < PARALLEL_MIN_MACS {
            1
        } else {
            threads
                .clamp(1, MAX_THREADS)
                .min(rows / MIN_ROWS_PER_WORKER)
        };
        if workers <= 1 {
            self.rows_into(path, &xr, first, count, &mut out);
            return out;
        }
        let per = rows.div_ceil(workers);
        std::thread::scope(|s| {
            for (xc, oc) in xr
                .chunks(per * self.in_pad)
                .zip(out.chunks_mut(per * count))
            {
                s.spawn(move || self.rows_into(path, xc, first, count, oc));
            }
        });
        out
    }

    /// Rotated rows `xr [T, in_pad]` against outputs `first..first + count` into `out`.
    fn rows_into(&self, path: Path, xr: &[f32], first: usize, count: usize, out: &mut [f32]) {
        match (path, self.bits) {
            #[cfg(target_arch = "aarch64")]
            (Path::Neon, 2) => self.rows_loop(
                xr,
                first,
                count,
                out,
                |x, o| self.dot_neon::<ROWS, 2>(x, o),
                |x, o| self.dot_neon::<1, 2>(x, o),
            ),
            #[cfg(target_arch = "aarch64")]
            (Path::Neon, _) => self.rows_loop(
                xr,
                first,
                count,
                out,
                |x, o| self.dot_neon::<ROWS, 4>(x, o),
                |x, o| self.dot_neon::<1, 4>(x, o),
            ),
            // SAFETY: `Path::Avx2` carries an `Avx2Detected`, which only `best_path` builds,
            // after `is_x86_feature_detected!("avx2")`.
            #[cfg(target_arch = "x86_64")]
            (Path::Avx2(_), _) => {
                debug_assert!(is_x86_feature_detected!("avx2"));
                unsafe { self.rows_into_avx2(xr, first, count, out) }
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            (Path::Simd128, 2) => self.rows_loop(
                xr,
                first,
                count,
                out,
                |x, o| self.dot_simd128::<ROWS, 2>(x, o),
                |x, o| self.dot_simd128::<1, 2>(x, o),
            ),
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            (Path::Simd128, _) => self.rows_loop(
                xr,
                first,
                count,
                out,
                |x, o| self.dot_simd128::<ROWS, 4>(x, o),
                |x, o| self.dot_simd128::<1, 4>(x, o),
            ),
            _ => self.rows_loop(
                xr,
                first,
                count,
                out,
                |x, o| self.dot_scalar::<ROWS>(x, o),
                |x, o| self.dot_scalar::<1>(x, o),
            ),
        }
    }

    /// The row loop of [`Self::rows_into`]: blocks of [`ROWS`] rows by `dot4`, the rest by
    /// `dot1`, each call one output. Inlined into every path, so a SIMD path compiled for its
    /// feature holds the whole loop.
    #[inline(always)]
    fn rows_loop(
        &self,
        xr: &[f32],
        first: usize,
        count: usize,
        out: &mut [f32],
        dot4: impl Fn([&[f32]; ROWS], usize) -> [f32; ROWS],
        dot1: impl Fn([&[f32]; 1], usize) -> [f32; 1],
    ) {
        let n = self.in_pad;
        let rows = xr.len() / n;
        let x = |r: usize| &xr[r * n..(r + 1) * n];
        let mut r = 0;
        while r + ROWS <= rows {
            let xs = [x(r), x(r + 1), x(r + 2), x(r + 3)];
            for c in 0..count {
                let o = dot4(xs, first + c);
                for (k, v) in o.iter().enumerate() {
                    out[(r + k) * count + c] = *v;
                }
            }
            r += ROWS;
        }
        // Two or three rows left (a batched decoder step of 2, 3, 6, 7, … hypotheses) share
        // one decode too: padded with zero rows to a block, whose outputs are dropped. Each
        // row's sum is its own, so the real rows get exactly the bits `dot1` gives them.
        if rows - r >= 2 {
            let zero = vec![0.0f32; n];
            let tail = |k: usize| if r + k < rows { x(r + k) } else { &zero[..] };
            let xs = [tail(0), tail(1), tail(2), tail(3)];
            for c in 0..count {
                let o = dot4(xs, first + c);
                for (k, v) in o.iter().enumerate().take(rows - r) {
                    out[(r + k) * count + c] = *v;
                }
            }
            r = rows;
        }
        while r < rows {
            for c in 0..count {
                out[r * count + c] = dot1([x(r)], first + c)[0];
            }
            r += 1;
        }
    }

    /// The portable order (module header).
    pub(crate) fn dot_scalar<const R: usize>(&self, xr: [&[f32]; R], o: usize) -> [f32; R] {
        let row = &self.cq.parts().packed[o * self.row_bytes..(o + 1) * self.row_bytes];
        let groups = self.in_pad / self.group;
        let cb = self.cq.parts().codebook;
        let mut out = [-0.0f32; R];
        for g in 0..groups {
            let mut acc = [[-0.0f32; LANES]; R];
            for k0 in (g * self.group..(g + 1) * self.group).step_by(LANES) {
                for j in 0..LANES {
                    let w = cb[code(row, k0 + j, self.bits)];
                    for (a, x) in acc.iter_mut().zip(&xr) {
                        a[j] += w * x[k0 + j];
                    }
                }
            }
            let norm = self.norms[o * groups + g];
            for (a, out) in acc.iter().zip(out.iter_mut()) {
                let v: [f32; 4] =
                    std::array::from_fn(|l| (a[l] + a[4 + l]) + (a[8 + l] + a[12 + l]));
                *out += norm * ((v[0] + v[1]) + (v[2] + v[3]));
            }
        }
        out
    }

    /// The codebook's bytes as four NEON registers (only the first is used at 2 bits).
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn lut_registers(&self) -> core::arch::aarch64::uint8x16x4_t {
        use core::arch::aarch64::*;
        // SAFETY: NEON is part of the aarch64 baseline; the four 16-byte loads read `lut`'s
        // 64 bytes.
        unsafe {
            uint8x16x4_t(
                vld1q_u8(self.lut.as_ptr()),
                vld1q_u8(self.lut.as_ptr().add(16)),
                vld1q_u8(self.lut.as_ptr().add(32)),
                vld1q_u8(self.lut.as_ptr().add(48)),
            )
        }
    }

    // Index loops over the fixed-size register arrays: they unroll fully and pair rows with lanes.
    #[allow(clippy::needless_range_loop)]
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn dot_neon<const R: usize, const B: u32>(&self, xr: [&[f32]; R], o: usize) -> [f32; R] {
        use core::arch::aarch64::*;
        let parts = self.cq.parts();
        let row = &parts.packed[o * self.row_bytes..(o + 1) * self.row_bytes];
        let groups = self.in_pad / self.group;
        assert!(row.len() * 8 >= self.in_pad * B as usize);
        assert!(xr.iter().all(|x| x.len() >= self.in_pad));
        let tables = DecodeTables::new::<B>();
        let mut out = [-0.0f32; R];
        // SAFETY: NEON is part of the aarch64 baseline. The index loads read 4 (`B` = 2) or 8
        // bytes at `k0 · B / 8` with `k0 + 16 ≤ in_pad`, within `row` (asserted
        // `in_pad · B / 8` bytes or more); the activation loads read 4 floats at
        // `k0 + 4v < in_pad`, within each `xr` row (asserted `in_pad` long); the store writes a
        // local 4-float array.
        unsafe {
            let lut = self.lut_registers();
            for g in 0..groups {
                let mut acc = [[vdupq_n_f32(-0.0); 4]; R];
                let mut k0 = g * self.group;
                while k0 < (g + 1) * self.group {
                    let w = tables.decode16::<B>(row.as_ptr().add(k0 * B as usize / 8), &lut);
                    for v in 0..4 {
                        for r in 0..R {
                            let x = vld1q_f32(xr[r].as_ptr().add(k0 + 4 * v));
                            acc[r][v] = vaddq_f32(acc[r][v], vmulq_f32(w[v], x));
                        }
                    }
                    k0 += LANES;
                }
                let norm = self.norms[o * groups + g];
                for r in 0..R {
                    let a = &acc[r];
                    let s = vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3]));
                    let mut v = [0.0f32; 4];
                    vst1q_f32(v.as_mut_ptr(), s);
                    out[r] += norm * ((v[0] + v[1]) + (v[2] + v[3]));
                }
            }
        }
        out
    }
}

/// The NEON index decode's shuffle constants for a width `B` (2 or 4).
#[cfg(target_arch = "aarch64")]
struct DecodeTables {
    /// Byte `p` of the 16 indices' source: 2-bit fields 4 to a byte, 4-bit 2 to a byte.
    rep: [u8; 16],
    /// Shifts each field to bits 2.., so the masked byte is `4 · code`, the first byte of the
    /// codebook entry: 2-bit field `l` sits at bit `2l`, 4-bit at `4l`.
    shift: [i8; 16],
    mask: u8,
    /// For output vector `v`: the 4 codes it takes, each spread over its 4 bytes.
    sel: [[u8; 16]; 4],
    /// The byte within each f32 lane.
    offs: [u8; 16],
}

#[cfg(target_arch = "aarch64")]
impl DecodeTables {
    #[inline(always)]
    fn new<const B: u32>() -> DecodeTables {
        DecodeTables {
            rep: std::array::from_fn(|p| (p * B as usize / 8) as u8),
            shift: std::array::from_fn(|p| 2 - (p * B as usize % 8) as i8),
            mask: (((1u32 << B) - 1) << 2) as u8,
            sel: std::array::from_fn(|v| std::array::from_fn(|b| (4 * v + b / 4) as u8)),
            offs: std::array::from_fn(|b| (b % 4) as u8),
        }
    }

    /// The codebook values of the 16 `B`-bit indices at `src`, as 4 vectors in index order.
    ///
    /// # Safety
    /// `src` must have 4 (`B` = 2) or 8 (`B` = 4) readable bytes; NEON is the aarch64 baseline.
    #[inline(always)]
    unsafe fn decode16<const B: u32>(
        &self,
        src: *const u8,
        lut: &core::arch::aarch64::uint8x16x4_t,
    ) -> [core::arch::aarch64::float32x4_t; 4] {
        use core::arch::aarch64::*;
        let raw = if B == 2 {
            vreinterpretq_u8_u32(vdupq_n_u32((src as *const u32).read_unaligned()))
        } else {
            vreinterpretq_u8_u64(vdupq_n_u64((src as *const u64).read_unaligned()))
        };
        let c4 = vandq_u8(
            vshlq_u8(
                vqtbl1q_u8(raw, vld1q_u8(self.rep.as_ptr())),
                vld1q_s8(self.shift.as_ptr()),
            ),
            vdupq_n_u8(self.mask),
        );
        let offs = vld1q_u8(self.offs.as_ptr());
        std::array::from_fn(|v| {
            let idx = vorrq_u8(vqtbl1q_u8(c4, vld1q_u8(self.sel[v].as_ptr())), offs);
            vreinterpretq_f32_u8(if B == 2 {
                vqtbl1q_u8(lut.0, idx)
            } else {
                vqtbl4q_u8(*lut, idx)
            })
        })
    }
}

#[path = "band.rs"]
mod band;

#[cfg(target_arch = "x86_64")]
#[path = "packed_x86.rs"]
mod x86;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[path = "packed_wasm.rs"]
mod wasm;

#[cfg(test)]
#[path = "packed_tests.rs"]
mod tests;
