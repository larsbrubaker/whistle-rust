//! The SSE2 (x86_64 baseline) lanes of `expf_lanes`: the scalar `expf`'s operations, four lanes
//! at a time, each branch chosen by selection (the parent module explains why the bits match).

use core::arch::x86_64::*;

use super::super::expf::{INV_LN2, LN2_HI, LN2_LO, P1, P2};
use super::{fix_special, scalar, BIG, MID, SPECIAL, TINY};

/// `mask ? a : b`, lane by lane, for an all-ones / all-zeros `mask`.
#[inline(always)]
fn select(mask: __m128i, a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: SSE2 is part of the x86_64 baseline.
    unsafe { _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b)) }
}

/// `expf` of each lane.
#[inline(always)]
fn exp4(x: __m128) -> __m128 {
    // SAFETY: SSE2 is part of the x86_64 baseline; the arrays the slow path stores to are
    // exactly four floats (unaligned stores and loads).
    unsafe {
        let bits = _mm_castps_si128(x);
        // |x| bits are non-negative as i32, so the signed compares order them as unsigned.
        let hx = _mm_and_si128(bits, _mm_set1_epi32(0x7fff_ffff));
        let neg = _mm_srai_epi32::<31>(bits);
        let special = _mm_cmpgt_epi32(hx, _mm_set1_epi32(SPECIAL as i32 - 1));
        let tiny = _mm_cmplt_epi32(hx, _mm_set1_epi32(TINY as i32 + 1));
        let big = _mm_cmpgt_epi32(hx, _mm_set1_epi32(BIG as i32));
        let mid = _mm_cmpgt_epi32(hx, _mm_set1_epi32(MID as i32));

        // k: trunc(INV_LN2 * x + ±0.5) when big, 1 - 2·sign when mid, else 0.
        let half = _mm_castsi128_ps(select(
            neg,
            _mm_castps_si128(_mm_set1_ps(-0.5)),
            _mm_castps_si128(_mm_set1_ps(0.5)),
        ));
        let kbig = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(INV_LN2), x), half));
        let kmid = select(neg, _mm_set1_epi32(-1), _mm_set1_epi32(1));
        let k = select(big, kbig, _mm_and_si128(mid, kmid));

        let kf = _mm_cvtepi32_ps(k);
        let hi = _mm_sub_ps(x, _mm_mul_ps(kf, _mm_set1_ps(LN2_HI)));
        let lo = _mm_mul_ps(kf, _mm_set1_ps(LN2_LO));
        let r = _mm_sub_ps(hi, lo);
        let rr = _mm_mul_ps(r, r);
        let poly = _mm_add_ps(_mm_set1_ps(P1), _mm_mul_ps(rr, _mm_set1_ps(P2)));
        let c = _mm_sub_ps(r, _mm_mul_ps(rr, poly));
        let one = _mm_set1_ps(1.0);
        let q = _mm_div_ps(_mm_mul_ps(r, c), _mm_sub_ps(_mm_set1_ps(2.0), c));
        let y = _mm_add_ps(one, _mm_add_ps(_mm_sub_ps(q, lo), hi));
        // 2^k from its exponent bits; k is in [-126, 126] in every lane that keeps this value.
        let scale = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(k, _mm_set1_epi32(0x7f))));
        let out = _mm_castsi128_ps(select(
            tiny,
            _mm_castps_si128(_mm_add_ps(one, x)),
            _mm_castps_si128(_mm_mul_ps(y, scale)),
        ));

        if _mm_movemask_ps(_mm_castsi128_ps(special)) == 0 {
            return out;
        }
        let (mut xs, mut ss, mut os) = ([0.0f32; 4], [0u32; 4], [0.0f32; 4]);
        _mm_storeu_ps(xs.as_mut_ptr(), x);
        _mm_storeu_si128(ss.as_mut_ptr().cast(), special);
        _mm_storeu_ps(os.as_mut_ptr(), out);
        fix_special(xs, ss, &mut os);
        _mm_loadu_ps(os.as_ptr())
    }
}

/// `1 / (1 + expf(-x))` of each lane.
#[inline(always)]
fn sigmoid4(x: __m128) -> __m128 {
    // SAFETY: SSE2 is part of the x86_64 baseline.
    unsafe {
        let one = _mm_set1_ps(1.0);
        // -x flips the sign bit, as the scalar negation does.
        let negx = _mm_xor_ps(x, _mm_set1_ps(-0.0));
        _mm_div_ps(one, _mm_add_ps(one, exp4(negx)))
    }
}

/// Applies `f` to each four-float chunk of `x` and `tail` to the rest.
#[inline(always)]
fn map(x: &mut [f32], f: impl Fn(__m128) -> __m128, tail: fn(&mut [f32])) {
    let mut chunks = x.chunks_exact_mut(4);
    for c in &mut chunks {
        // SAFETY: `c` is exactly four floats (unaligned load and store); SSE2 is baseline.
        unsafe { _mm_storeu_ps(c.as_mut_ptr(), f(_mm_loadu_ps(c.as_ptr()))) }
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
    // SAFETY: SSE2 is part of the x86_64 baseline.
    map(
        x,
        |v| unsafe { _mm_mul_ps(v, sigmoid4(v)) },
        scalar::silu_in_place,
    );
}
