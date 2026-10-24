//! The NEON (aarch64) lanes of `expf_lanes`: the scalar `expf`'s operations, four lanes at a
//! time, each branch chosen by selection (the parent module explains why the bits match).

use core::arch::aarch64::*;

use super::super::expf::{INV_LN2, LN2_HI, LN2_LO, P1, P2};
use super::{fix_special, scalar, BIG, MID, SPECIAL, TINY};

/// `expf` of each lane.
// SAFETY (whole function): NEON is part of the aarch64 baseline, so its intrinsics are always
// available; the arrays the slow path stores to are exactly four floats.
#[inline(always)]
fn exp4(x: float32x4_t) -> float32x4_t {
    unsafe {
        let bits = vreinterpretq_u32_f32(x);
        let hx = vandq_u32(bits, vdupq_n_u32(0x7fff_ffff));
        // All ones in the lanes where x is negative (the scalar's `sign`).
        let neg = vreinterpretq_u32_s32(vshrq_n_s32::<31>(vreinterpretq_s32_u32(bits)));
        let special = vcgeq_u32(hx, vdupq_n_u32(SPECIAL));
        let tiny = vcleq_u32(hx, vdupq_n_u32(TINY));
        let big = vcgtq_u32(hx, vdupq_n_u32(BIG));
        let mid = vcgtq_u32(hx, vdupq_n_u32(MID));

        // k: trunc(INV_LN2 * x + ±0.5) when big, 1 - 2·sign when mid, else 0.
        let half = vbslq_f32(neg, vdupq_n_f32(-0.5), vdupq_n_f32(0.5));
        let kbig = vcvtq_s32_f32(vaddq_f32(vmulq_f32(vdupq_n_f32(INV_LN2), x), half));
        let kmid = vbslq_s32(neg, vdupq_n_s32(-1), vdupq_n_s32(1));
        let k = vbslq_s32(big, kbig, vandq_s32(vreinterpretq_s32_u32(mid), kmid));

        let kf = vcvtq_f32_s32(k);
        let hi = vsubq_f32(x, vmulq_f32(kf, vdupq_n_f32(LN2_HI)));
        let lo = vmulq_f32(kf, vdupq_n_f32(LN2_LO));
        let r = vsubq_f32(hi, lo);
        let rr = vmulq_f32(r, r);
        let poly = vaddq_f32(vdupq_n_f32(P1), vmulq_f32(rr, vdupq_n_f32(P2)));
        let c = vsubq_f32(r, vmulq_f32(rr, poly));
        let one = vdupq_n_f32(1.0);
        let q = vdivq_f32(vmulq_f32(r, c), vsubq_f32(vdupq_n_f32(2.0), c));
        let y = vaddq_f32(one, vaddq_f32(vsubq_f32(q, lo), hi));
        // 2^k from its exponent bits; k is in [-126, 126] in every lane that keeps this value.
        let scale = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(k, vdupq_n_s32(0x7f))));
        let out = vbslq_f32(tiny, vaddq_f32(one, x), vmulq_f32(y, scale));

        if vmaxvq_u32(special) == 0 {
            return out;
        }
        let (mut xs, mut ss, mut os) = ([0.0f32; 4], [0u32; 4], [0.0f32; 4]);
        vst1q_f32(xs.as_mut_ptr(), x);
        vst1q_u32(ss.as_mut_ptr(), special);
        vst1q_f32(os.as_mut_ptr(), out);
        fix_special(xs, ss, &mut os);
        vld1q_f32(os.as_ptr())
    }
}

/// `1 / (1 + expf(-x))` of each lane.
#[inline(always)]
fn sigmoid4(x: float32x4_t) -> float32x4_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let one = vdupq_n_f32(1.0);
        vdivq_f32(one, vaddq_f32(one, exp4(vnegq_f32(x))))
    }
}

/// Applies `f` to each four-float chunk of `x` and `tail` to the rest.
#[inline(always)]
fn map(x: &mut [f32], f: impl Fn(float32x4_t) -> float32x4_t, tail: fn(&mut [f32])) {
    let mut chunks = x.chunks_exact_mut(4);
    for c in &mut chunks {
        // SAFETY: `c` is exactly four floats; NEON is part of the aarch64 baseline.
        unsafe { vst1q_f32(c.as_mut_ptr(), f(vld1q_f32(c.as_ptr()))) }
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
    // SAFETY: NEON is part of the aarch64 baseline.
    map(
        x,
        |v| unsafe { vmulq_f32(v, sigmoid4(v)) },
        scalar::silu_in_place,
    );
}
