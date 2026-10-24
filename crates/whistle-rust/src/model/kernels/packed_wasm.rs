//! The wasm `simd128` paths of both packed kernels (`kernels::packed`): the rotated projection
//! and the band decode, each bit-identical to the scalar order by construction. Compiled only
//! with `target_feature = "simd128"` (a compile-time feature on wasm), so every function here
//! inlines into the row and band loops of `packed` and `band`.
//!
//! **Index decode** (16 fields, one code per byte). wasm has no per-lane variable shift, so the
//! fields are split by nibble:
//! - 2-bit: the 4 source bytes are spread so byte `p` holds source byte `p / 4`
//!   (`i8x16_swizzle`). The low nibble (`& 15`) serves positions `p % 4 < 2`, the high nibble
//!   (`>> 4`) positions 2 and 3, picked by `v128_bitselect`. Then an even `p` takes `nib & 3`
//!   and an odd one `nib >> 2`: code `k0 + p`.
//! - 4-bit: byte `p` holds source byte `p / 2`; an even `p` takes the low nibble and an odd one
//!   the high nibble.
//!
//! **Value lookup.** The 16-entry codebook as four byte planes (plane `b` = byte `b` of every
//! entry), so `i8x16_swizzle(plane_b, codes)` is byte `b` of all 16 values. Two rounds of byte
//! interleaves (`i8x16_shuffle`) rebuild the little-endian f32s: the first pairs planes 0/1 and
//! 2/3 by value, the second joins the pairs. Vector `v` then holds values `4v..4v + 4`.
//!
//! **Lane mapping** is NEON's: vector `v`, lane `l` is the scalar order's lane `4v + l`. The
//! reduction `(acc0 + acc1) + (acc2 + acc3)` gives `v_l = (a_l + a_{4+l}) + (a_{8+l} +
//! a_{12+l})`, then `(v_0 + v_1) + (v_2 + v_3)` in scalar. `f32x4_mul` then `f32x4_add`; wasm
//! has no FMA in simd128.
//!
//! **Butterflies** (the band decode), on `a = [x0..x3]`, `b = [x4..x7]`:
//! - Level 1: `i32x4_shuffle` `<0,2,4,6>` and `<1,3,5,7>` give `[x0,x2,x4,x6]` and
//!   `[x1,x3,x5,x7]`. With the sum `s` and difference `d`, `<0,4,1,5>(s, d)` is
//!   `[x0+x1, x0−x1, x2+x3, x2−x3]` and `<2,6,3,7>(s, d)` is the same for `b`.
//! - Level 2: `<0,1,4,5>` and `<2,3,6,7>` give `[x0,x1,x4,x5]` and `[x2,x3,x6,x7]`. Then
//!   `<0,1,4,5>(s, d)` is `[x0+x2, x1+x3, x0−x2, x1−x3]` and `<2,3,6,7>(s, d)` is `b`'s.
//! - Levels 4 and 8 add and subtract whole vectors; higher levels are 4-wide in memory.
//!
//! Every element is `x_lo + x_hi` or `x_lo − x_hi`, as in `quant::walsh_hadamard`.

use core::arch::wasm32::*;

use super::PackedMatrix;

/// The byte planes of a 16-entry f32 table held as 4 vectors (entries `4i..4i + 4` in `e[i]`).
#[inline(always)]
fn planes(e: [v128; 4]) -> [v128; 4] {
    // Bytes `b, 4 + b, …` of two vectors: one plane's 8 entries, for `b` in 0, 1 (then 2, 3).
    let p01 =
        |x, y| i8x16_shuffle::<0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29>(x, y);
    let p23 =
        |x, y| i8x16_shuffle::<2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31>(x, y);
    let (a01, a23) = (p01(e[0], e[1]), p23(e[0], e[1]));
    let (b01, b23) = (p01(e[2], e[3]), p23(e[2], e[3]));
    [
        i64x2_shuffle::<0, 2>(a01, b01),
        i64x2_shuffle::<1, 3>(a01, b01),
        i64x2_shuffle::<0, 2>(a23, b23),
        i64x2_shuffle::<1, 3>(a23, b23),
    ]
}

/// The 16 codes at `src`, one per byte, in order.
///
/// # Safety
/// `src` must have 4 (`B` = 2) or 8 (`B` = 4) readable bytes.
#[inline(always)]
unsafe fn codes16<const B: u32>(src: *const u8) -> v128 {
    let odd = i8x16(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1);
    if B == 2 {
        let raw = u32x4_splat((src as *const u32).read_unaligned());
        let rep = i8x16_swizzle(raw, i8x16(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3));
        let upper = i8x16(0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1);
        let nib = v128_bitselect(u8x16_shr(rep, 4), v128_and(rep, u8x16_splat(15)), upper);
        v128_bitselect(u8x16_shr(nib, 2), v128_and(nib, u8x16_splat(3)), odd)
    } else {
        let raw = u64x2_splat((src as *const u64).read_unaligned());
        let rep = i8x16_swizzle(raw, i8x16(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7));
        v128_bitselect(u8x16_shr(rep, 4), v128_and(rep, u8x16_splat(15)), odd)
    }
}

/// The table values of 16 codes (one per byte) through byte planes: 4 vectors in code order.
#[inline(always)]
fn lookup16(codes: v128, planes: &[v128; 4]) -> [v128; 4] {
    let b = planes.map(|p| i8x16_swizzle(p, codes));
    let lo = |x, y| i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(x, y);
    let hi =
        |x, y| i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(x, y);
    let (b01l, b01h, b23l, b23h) = (
        lo(b[0], b[1]),
        hi(b[0], b[1]),
        lo(b[2], b[3]),
        hi(b[2], b[3]),
    );
    let first =
        |x, y| i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(x, y);
    let second =
        |x, y| i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(x, y);
    [
        first(b01l, b23l),
        second(b01l, b23l),
        first(b01h, b23h),
        second(b01h, b23h),
    ]
}

impl PackedMatrix {
    /// The codebook as 4 vectors of 4 entries (zero past a 2-bit codebook's 4).
    #[inline(always)]
    fn table(&self) -> [v128; 4] {
        // SAFETY: `cbf` is 16 floats; load `i` reads floats `4i..4i + 4`; `v128_load` has no
        // alignment requirement.
        std::array::from_fn(|i| unsafe { v128_load(self.cbf.as_ptr().add(4 * i) as *const v128) })
    }

    /// Output `o` for `R` rotated rows (`packed::rows_into`).
    // Index loops over the fixed-size register arrays: they unroll fully and pair rows with lanes.
    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    pub(super) fn dot_simd128<const R: usize, const B: u32>(
        &self,
        xr: [&[f32]; R],
        o: usize,
    ) -> [f32; R] {
        let row = &self.cq.parts().packed[o * self.row_bytes..(o + 1) * self.row_bytes];
        let groups = self.in_pad / self.group;
        assert!(row.len() * 8 >= self.in_pad * B as usize);
        assert!(xr.iter().all(|x| x.len() >= self.in_pad));
        let planes = planes(self.table());
        let mut out = [-0.0f32; R];
        // SAFETY: simd128 is enabled at compile time. The index loads read 4 or 8 bytes at
        // `k0 · B / 8`, `k0 + 16 ≤ in_pad`, within `row` (asserted); the activation loads read
        // 4 floats at `k0 + 4v < in_pad` within each `xr` row (asserted); the store writes a
        // local 4-float array.
        unsafe {
            for g in 0..groups {
                let mut acc = [[f32x4_splat(-0.0); 4]; R];
                for k0 in (g * self.group..(g + 1) * self.group).step_by(16) {
                    let codes = codes16::<B>(row.as_ptr().add(k0 * B as usize / 8));
                    let w = lookup16(codes, &planes);
                    for v in 0..4 {
                        for r in 0..R {
                            let x = v128_load(xr[r].as_ptr().add(k0 + 4 * v) as *const v128);
                            acc[r][v] = f32x4_add(acc[r][v], f32x4_mul(w[v], x));
                        }
                    }
                }
                let norm = self.norms[o * groups + g];
                for r in 0..R {
                    let a = &acc[r];
                    let s = f32x4_add(f32x4_add(a[0], a[1]), f32x4_add(a[2], a[3]));
                    let v = [
                        f32x4_extract_lane::<0>(s),
                        f32x4_extract_lane::<1>(s),
                        f32x4_extract_lane::<2>(s),
                        f32x4_extract_lane::<3>(s),
                    ];
                    out[r] += norm * ((v[0] + v[1]) + (v[2] + v[3]));
                }
            }
        }
        out
    }

    /// Group `g` of a row dequantized in place: `(cb · norm) · scale` by table, then the
    /// butterflies (module header).
    #[inline(always)]
    pub(super) fn decode_group_simd128<const B: u32>(
        &self,
        row: &[u8],
        g: usize,
        norm: f32,
        chunk: &mut [f32],
    ) {
        let n = self.group;
        // The butterflies need a power-of-two group (`PackedMatrix::new` admits no other).
        assert!(chunk.len() == n && n.is_power_of_two() && n >= 16);
        assert!(row.len() * 8 >= (g + 1) * n * B as usize);
        // The same scale `walsh_hadamard` multiplies by.
        let scale = (1.0 / (n as f64).sqrt()) as f32;
        let (vn, vs) = (f32x4_splat(norm), f32x4_splat(scale));
        let planes = planes(self.table().map(|e| f32x4_mul(f32x4_mul(e, vn), vs)));
        let half1 = |a, b| {
            let (t1, t2) = (
                i32x4_shuffle::<0, 2, 4, 6>(a, b),
                i32x4_shuffle::<1, 3, 5, 7>(a, b),
            );
            let (s, d) = (f32x4_add(t1, t2), f32x4_sub(t1, t2));
            (
                i32x4_shuffle::<0, 4, 1, 5>(s, d),
                i32x4_shuffle::<2, 6, 3, 7>(s, d),
            )
        };
        let half2 = |a, b| {
            let (t1, t2) = (
                i32x4_shuffle::<0, 1, 4, 5>(a, b),
                i32x4_shuffle::<2, 3, 6, 7>(a, b),
            );
            let (s, d) = (f32x4_add(t1, t2), f32x4_sub(t1, t2));
            (
                i32x4_shuffle::<0, 1, 4, 5>(s, d),
                i32x4_shuffle::<2, 3, 6, 7>(s, d),
            )
        };
        let p = chunk.as_mut_ptr();
        // SAFETY: simd128 is enabled at compile time. The index loads read 4 or 8 bytes at
        // `(g · n + k0) · B / 8`, `k0 + 16 ≤ n`, within `row` (asserted); the float loads and
        // stores touch `chunk[i..i + 4]`, `i + 4 ≤ n` (asserted `n` long): the group `n` is a
        // power of two of at least 16 (asserted), so `16 | n` and, at every memory level
        // `half ≥ 16`, `2 · half` divides `n` and each partner `i + half < n`. `v128_load` and
        // `v128_store` have no alignment requirement.
        unsafe {
            for k0 in (0..n).step_by(16) {
                let codes = codes16::<B>(row.as_ptr().add((g * n + k0) * B as usize / 8));
                let w = lookup16(codes, &planes);
                let (a0, a1) = half1(w[0], w[1]);
                let (a2, a3) = half1(w[2], w[3]);
                let (b0, b1) = half2(a0, a1);
                let (b2, b3) = half2(a2, a3);
                let (c0, c1) = (f32x4_add(b0, b1), f32x4_sub(b0, b1));
                let (c2, c3) = (f32x4_add(b2, b3), f32x4_sub(b2, b3));
                let q = p.add(k0) as *mut v128;
                v128_store(q, f32x4_add(c0, c2));
                v128_store(q.add(1), f32x4_add(c1, c3));
                v128_store(q.add(2), f32x4_sub(c0, c2));
                v128_store(q.add(3), f32x4_sub(c1, c3));
            }
            let mut half = 16;
            while half < n {
                for block in (0..n).step_by(2 * half) {
                    for i in (block..block + half).step_by(4) {
                        let (x, y) = (p.add(i) as *mut v128, p.add(i + half) as *mut v128);
                        let (a, b) = (v128_load(x), v128_load(y));
                        v128_store(x, f32x4_add(a, b));
                        v128_store(y, f32x4_sub(a, b));
                    }
                }
                half *= 2;
            }
        }
    }
}
