//! The AVX2 paths of both packed kernels (`kernels::packed`): the rotated projection and the
//! band decode, each bit-identical to the scalar order by construction. They are compiled for
//! AVX2 and reached only through `Path::Avx2`, whose `Avx2Detected` token proves the feature was
//! detected. Each entry point ([`PackedMatrix::rows_into_avx2`],
//! [`PackedMatrix::band_into_avx2`]) holds the whole per-projection or per-band loop, so the
//! group code inlines into it: there is one feature-enabled call per projection or band, never
//! one per group.
//!
//! **Index decode.** 16 fields at once: 4 bytes (2-bit) or 8 bytes (4-bit), read unaligned and
//! broadcast to 8 `u32` lanes. A per-lane variable shift (`_mm256_srlv_epi32`) and a mask leave
//! code `k0 + j` in lane `j` of the first vector and code `k0 + 8 + j` in lane `j` of the second.
//! For 2 bits that is shift `2j` (`2j + 16`); for 4 bits, shift `4j` of the low and the high
//! `u32`. The codebook value is `_mm256_permutevar8x32_ps(table, code)`, which takes lane
//! `code & 7` of an 8-float table. At 4 bits a second lookup into entries 8–15 and
//! `_mm256_blendv_ps` on bit 3 (shifted to the sign bit) pick the entry.
//!
//! **Lane mapping.** The scalar order's lane `j` (inputs `k ≡ j mod 16`) is lane `j` of the
//! first accumulator for `j < 8` and lane `j − 8` of the second. Its reduction
//! `v_l = (a_l + a_{4+l}) + (a_{8+l} + a_{12+l})` is (low half + high half of the first) +
//! (low half + high half of the second), then `(v_0 + v_1) + (v_2 + v_3)` in scalar. Products
//! are `_mm256_mul_ps` then `_mm256_add_ps`, never FMA.
//!
//! **Butterflies** (the band decode), on the 16 values as two vectors `[x0..x7]`, `[x8..x15]`.
//! Every element is `x_lo + x_hi` or `x_lo − x_hi`, as in `quant::walsh_hadamard`:
//! - Level 1: `_mm256_shuffle_ps` with 0xA0 and 0xF5 give `[x0,x0,x2,x2]` and `[x1,x1,x3,x3]`
//!   per 128-bit half. Add and subtract, then `_mm256_blend_ps` 0xAA (odd lanes from the
//!   difference) gives `[x0+x1, x0−x1, x2+x3, x2−x3]`.
//! - Level 2: 0x44 and 0xEE give `[x0,x1,x0,x1]` and `[x2,x3,x2,x3]`. Blend 0xCC gives
//!   `[x0+x2, x1+x3, x0−x2, x1−x3]`.
//! - Level 4: `_mm256_permute2f128_ps` 0x00 and 0x11 give `[L, L]` and `[H, H]`. Blend 0xF0
//!   gives `[L+H, L−H]`.
//! - Level 8 is the two vectors' sum and difference; higher levels are 8-wide in memory.

use core::arch::x86_64::*;

use super::{PackedMatrix, ROWS};

/// The 16 codebook values of the fields at `src`, codes `k0..k0 + 8` then `k0 + 8..k0 + 16`.
///
/// # Safety
/// AVX2 must be available; `src` must have 4 (`B` = 2) or 8 (`B` = 4) readable bytes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn decode16<const B: u32>(src: *const u8, t0: __m256, t1: __m256) -> [__m256; 2] {
    let (lo, hi) = if B == 2 {
        let w = _mm256_set1_epi32((src as *const u32).read_unaligned() as i32);
        let mask = _mm256_set1_epi32(3);
        (
            _mm256_and_si256(
                _mm256_srlv_epi32(w, _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14)),
                mask,
            ),
            _mm256_and_si256(
                _mm256_srlv_epi32(w, _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30)),
                mask,
            ),
        )
    } else {
        let w = (src as *const u64).read_unaligned();
        let shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
        let mask = _mm256_set1_epi32(15);
        (
            _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32(w as u32 as i32), shifts),
                mask,
            ),
            _mm256_and_si256(
                _mm256_srlv_epi32(_mm256_set1_epi32((w >> 32) as u32 as i32), shifts),
                mask,
            ),
        )
    };
    let look = |c: __m256i| {
        let low = _mm256_permutevar8x32_ps(t0, c);
        if B == 2 {
            low
        } else {
            // Bit 3 of the code, moved to the sign bit, selects entries 8–15.
            let high = _mm256_permutevar8x32_ps(t1, c);
            _mm256_blendv_ps(low, high, _mm256_castsi256_ps(_mm256_slli_epi32(c, 28)))
        }
    };
    [look(lo), look(hi)]
}

impl PackedMatrix {
    /// The rotated kernel's row loop on AVX2 (`packed::rows_into`).
    ///
    /// # Safety
    /// AVX2 must be available (the caller holds an `Avx2Detected`).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn rows_into_avx2(
        &self,
        xr: &[f32],
        first: usize,
        count: usize,
        out: &mut [f32],
    ) {
        if self.bits == 2 {
            self.rows_loop(
                xr,
                first,
                count,
                out,
                |x, o| self.dot_avx2::<ROWS, 2>(x, o),
                |x, o| self.dot_avx2::<1, 2>(x, o),
            );
        } else {
            self.rows_loop(
                xr,
                first,
                count,
                out,
                |x, o| self.dot_avx2::<ROWS, 4>(x, o),
                |x, o| self.dot_avx2::<1, 4>(x, o),
            );
        }
    }

    /// Output `o` for `R` rotated rows: the scalar order's lanes in two 8-lane accumulators.
    // Index loops over the fixed-size register arrays: they unroll fully and pair rows with lanes.
    #[allow(clippy::needless_range_loop)]
    #[target_feature(enable = "avx2")]
    #[inline]
    fn dot_avx2<const R: usize, const B: u32>(&self, xr: [&[f32]; R], o: usize) -> [f32; R] {
        let row = &self.cq.parts().packed[o * self.row_bytes..(o + 1) * self.row_bytes];
        let groups = self.in_pad / self.group;
        assert!(row.len() * 8 >= self.in_pad * B as usize);
        assert!(xr.iter().all(|x| x.len() >= self.in_pad));
        let mut out = [-0.0f32; R];
        // SAFETY: AVX2 is enabled on this function. The table loads read `cbf`'s 16 floats; the
        // index loads read 4 (`B` = 2) or 8 bytes at `k0 · B / 8`, `k0 + 16 ≤ in_pad`, within
        // `row` (asserted); the activation loads read 8 floats at `k0` and `k0 + 8`, within
        // each `xr` row (asserted `in_pad` long); the store writes a local 4-float array.
        unsafe {
            let t0 = _mm256_loadu_ps(self.cbf.as_ptr());
            let t1 = _mm256_loadu_ps(self.cbf.as_ptr().add(8));
            for g in 0..groups {
                let mut acc = [[_mm256_set1_ps(-0.0); 2]; R];
                for k0 in (g * self.group..(g + 1) * self.group).step_by(16) {
                    let w = decode16::<B>(row.as_ptr().add(k0 * B as usize / 8), t0, t1);
                    for r in 0..R {
                        let p = xr[r].as_ptr().add(k0);
                        let (x0, x1) = (_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8)));
                        acc[r][0] = _mm256_add_ps(acc[r][0], _mm256_mul_ps(w[0], x0));
                        acc[r][1] = _mm256_add_ps(acc[r][1], _mm256_mul_ps(w[1], x1));
                    }
                }
                let norm = self.norms[o * groups + g];
                for r in 0..R {
                    let [a, b] = acc[r];
                    let ta = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
                    let tb = _mm_add_ps(_mm256_castps256_ps128(b), _mm256_extractf128_ps(b, 1));
                    let mut v = [0.0f32; 4];
                    _mm_storeu_ps(v.as_mut_ptr(), _mm_add_ps(ta, tb));
                    out[r] += norm * ((v[0] + v[1]) + (v[2] + v[3]));
                }
            }
        }
        out
    }

    /// The band decode's loop on AVX2 (`band::band_into`).
    ///
    /// # Safety
    /// AVX2 must be available (the caller holds an `Avx2Detected`).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn band_into_avx2(&self, first: usize, out: &mut [f32], padded: &mut [f32]) {
        if self.bits == 2 {
            self.band_loop(first, out, padded, |row, g, norm, chunk| {
                self.decode_group_avx2::<2>(row, g, norm, chunk)
            });
        } else {
            self.band_loop(first, out, padded, |row, g, norm, chunk| {
                self.decode_group_avx2::<4>(row, g, norm, chunk)
            });
        }
    }

    /// Group `g` of a row dequantized in place: `(cb · norm) · scale` by table, then the
    /// butterflies (module header).
    #[target_feature(enable = "avx2")]
    #[inline]
    fn decode_group_avx2<const B: u32>(&self, row: &[u8], g: usize, norm: f32, chunk: &mut [f32]) {
        let n = self.group;
        // The butterflies need a power-of-two group (`PackedMatrix::new` admits no other).
        assert!(chunk.len() == n && n.is_power_of_two() && n >= 16);
        assert!(row.len() * 8 >= (g + 1) * n * B as usize);
        // The same scale `walsh_hadamard` multiplies by.
        let scale = (1.0 / (n as f64).sqrt()) as f32;
        // SAFETY: AVX2 is enabled on this function. The table loads read `cbf`'s 16 floats; the
        // index loads read 4 or 8 bytes at `(g · n + k0) · B / 8`, `k0 + 16 ≤ n`, within `row`
        // (asserted); the float loads and stores touch `chunk[i..i + 8]`, `i + 8 ≤ n`: the group
        // `n` is a power of two of at least 16 (asserted), so `16 | n` and, at every memory
        // level `half ≥ 16`, `2 · half` divides `n` and each partner `i + half < n`.
        unsafe {
            let (vn, vs) = (_mm256_set1_ps(norm), _mm256_set1_ps(scale));
            let fold = |p: *const f32| _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(p), vn), vs);
            let (t0, t1) = (fold(self.cbf.as_ptr()), fold(self.cbf.as_ptr().add(8)));
            let butterfly = |x: __m256| {
                let e = _mm256_shuffle_ps::<0xA0>(x, x);
                let o = _mm256_shuffle_ps::<0xF5>(x, x);
                let x = _mm256_blend_ps::<0xAA>(_mm256_add_ps(e, o), _mm256_sub_ps(e, o));
                let e = _mm256_shuffle_ps::<0x44>(x, x);
                let o = _mm256_shuffle_ps::<0xEE>(x, x);
                let x = _mm256_blend_ps::<0xCC>(_mm256_add_ps(e, o), _mm256_sub_ps(e, o));
                let l = _mm256_permute2f128_ps::<0x00>(x, x);
                let h = _mm256_permute2f128_ps::<0x11>(x, x);
                _mm256_blend_ps::<0xF0>(_mm256_add_ps(l, h), _mm256_sub_ps(l, h))
            };
            let p = chunk.as_mut_ptr();
            for k0 in (0..n).step_by(16) {
                let src = row.as_ptr().add((g * n + k0) * B as usize / 8);
                let [w0, w1] = decode16::<B>(src, t0, t1);
                let (a, b) = (butterfly(w0), butterfly(w1));
                _mm256_storeu_ps(p.add(k0), _mm256_add_ps(a, b));
                _mm256_storeu_ps(p.add(k0 + 8), _mm256_sub_ps(a, b));
            }
            let mut half = 16;
            while half < n {
                for block in (0..n).step_by(2 * half) {
                    for i in (block..block + half).step_by(8) {
                        let (x, y) = (_mm256_loadu_ps(p.add(i)), _mm256_loadu_ps(p.add(i + half)));
                        _mm256_storeu_ps(p.add(i), _mm256_add_ps(x, y));
                        _mm256_storeu_ps(p.add(i + half), _mm256_sub_ps(x, y));
                    }
                }
                half *= 2;
            }
        }
    }
}
