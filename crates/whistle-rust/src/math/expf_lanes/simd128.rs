//! The wasm `simd128` lanes of `expf_lanes`: the scalar `expf`'s operations, four lanes at a
//! time, each branch chosen by selection (the parent module explains why the bits match).

use core::arch::wasm32::*;

use super::super::expf::{INV_LN2, LN2_HI, LN2_LO, P1, P2};
use super::{fix_special, scalar, BIG, MID, SPECIAL, TINY};

/// `expf` of each lane.
#[inline(always)]
fn exp4(x: v128) -> v128 {
    let hx = v128_and(x, u32x4_splat(0x7fff_ffff));
    // All ones in the lanes where x is negative (the scalar's `sign`).
    let neg = i32x4_shr(x, 31);
    let special = u32x4_ge(hx, u32x4_splat(SPECIAL));
    let tiny = u32x4_le(hx, u32x4_splat(TINY));
    let big = u32x4_gt(hx, u32x4_splat(BIG));
    let mid = u32x4_gt(hx, u32x4_splat(MID));

    // k: trunc(INV_LN2 * x + ±0.5) when big, 1 - 2·sign when mid, else 0.
    let half = v128_bitselect(f32x4_splat(-0.5), f32x4_splat(0.5), neg);
    let kbig = i32x4_trunc_sat_f32x4(f32x4_add(f32x4_mul(f32x4_splat(INV_LN2), x), half));
    let kmid = v128_bitselect(i32x4_splat(-1), i32x4_splat(1), neg);
    let k = v128_bitselect(kbig, v128_and(mid, kmid), big);

    let kf = f32x4_convert_i32x4(k);
    let hi = f32x4_sub(x, f32x4_mul(kf, f32x4_splat(LN2_HI)));
    let lo = f32x4_mul(kf, f32x4_splat(LN2_LO));
    let r = f32x4_sub(hi, lo);
    let rr = f32x4_mul(r, r);
    let poly = f32x4_add(f32x4_splat(P1), f32x4_mul(rr, f32x4_splat(P2)));
    let c = f32x4_sub(r, f32x4_mul(rr, poly));
    let one = f32x4_splat(1.0);
    let q = f32x4_div(f32x4_mul(r, c), f32x4_sub(f32x4_splat(2.0), c));
    let y = f32x4_add(one, f32x4_add(f32x4_sub(q, lo), hi));
    // 2^k from its exponent bits; k is in [-126, 126] in every lane that keeps this value.
    let scale = i32x4_shl(i32x4_add(k, i32x4_splat(0x7f)), 23);
    let out = v128_bitselect(f32x4_add(one, x), f32x4_mul(y, scale), tiny);

    if !v128_any_true(special) {
        return out;
    }
    let lanes_f = |v: v128| {
        [
            f32x4_extract_lane::<0>(v),
            f32x4_extract_lane::<1>(v),
            f32x4_extract_lane::<2>(v),
            f32x4_extract_lane::<3>(v),
        ]
    };
    let ss = [
        u32x4_extract_lane::<0>(special),
        u32x4_extract_lane::<1>(special),
        u32x4_extract_lane::<2>(special),
        u32x4_extract_lane::<3>(special),
    ];
    let mut os = lanes_f(out);
    fix_special(lanes_f(x), ss, &mut os);
    f32x4(os[0], os[1], os[2], os[3])
}

/// `1 / (1 + expf(-x))` of each lane.
#[inline(always)]
fn sigmoid4(x: v128) -> v128 {
    let one = f32x4_splat(1.0);
    f32x4_div(one, f32x4_add(one, exp4(f32x4_neg(x))))
}

/// Applies `f` to each four-float chunk of `x` and `tail` to the rest.
#[inline(always)]
fn map(x: &mut [f32], f: impl Fn(v128) -> v128, tail: fn(&mut [f32])) {
    let mut chunks = x.chunks_exact_mut(4);
    for c in &mut chunks {
        // SAFETY: `c` is exactly four floats; v128 loads and stores need no alignment.
        unsafe { v128_store(c.as_mut_ptr().cast(), f(v128_load(c.as_ptr().cast()))) }
    }
    tail(chunks.into_remainder());
}

pub(super) fn exp_in_place(x: &mut [f32]) {
    map(x, exp4, scalar::exp_in_place);
}

pub(super) fn sigmoid_in_place(x: &mut [f32]) {
    map(x, sigmoid4, scalar::sigmoid_in_place);
}

pub(super) fn silu_in_place(x: &mut [f32]) {
    map(x, |v| f32x4_mul(v, sigmoid4(v)), scalar::silu_in_place);
}
