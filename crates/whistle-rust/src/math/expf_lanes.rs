//! `expf`, `sigmoid` and `silu` over f32 slices, in place, four lanes at a time, giving exactly
//! the bits of the scalar `expf` (`expf.rs`) and of `1 / (1 + expf(-x))` and
//! `x * (1 / (1 + expf(-x)))` in every element. Part of `math`; the model's activations call it
//! (`model::ops`).
//!
//! **Why the bits match.** Each lane does the scalar code's IEEE operations in the scalar code's
//! order: the same multiplies, adds, subtracts and the one division, no fused multiply-add, no
//! approximate reciprocal, nothing reassociated. The scalar code branches on `|x|`; here every
//! branch of the common range is computed by selection instead:
//! - `|x| ≤ 2⁻¹⁴`: `1 + x`;
//! - `2⁻¹⁴ < |x| ≤ ½ln2`: `k = 0` (then `hi = x − 0·ln2hi = x` and `lo = 0` exactly, which
//!   is the scalar's `hi = x, lo = 0`);
//! - `½ln2 < |x| ≤ 1.5ln2`: `k = ±1`;
//! - `1.5ln2 < |x| < 87.34`: `k = trunc(invln2·x ± ½)`, the scalar's `as i32` (truncation,
//!   exact for these magnitudes).
//!
//! In that range `k` stays in `[−126, 126]`, where the scalar's `scalbnf(y, k)` is the single
//! multiply `y · 2ᵏ` with `2ᵏ` built from its exponent bits, so the lanes do that multiply.
//! Lanes with `|x| ≥ 87.34` or NaN (overflow, underflow, infinities, and `scalbnf`'s multi-step
//! scalings) are recomputed with the scalar `expf`. Vector `+ − × ÷` round as the scalar ones
//! do and keep subnormals on every target here (AArch64 Advanced SIMD follows the FPCR, which
//! Rust leaves IEEE; SSE2 without FTZ/DAZ; wasm SIMD is IEEE), so the selection is the only
//! difference. `tests.rs` checks every lane against the scalar code, all 2³² inputs in an
//! ignored test.
//!
//! The vector paths are chosen at compile time: NEON on aarch64, SSE2 (the x86_64 baseline),
//! `simd128` on wasm32 built with it, otherwise the scalar code itself.

// Without a vector path only the scalar module and the tests use the range bounds and the
// slow-path helper.
#![cfg_attr(
    not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128;
#[cfg(target_arch = "x86_64")]
mod sse2;
#[cfg(test)]
mod tests;

#[cfg(target_arch = "aarch64")]
use neon as lanes;
#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
use scalar as lanes;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use simd128 as lanes;
#[cfg(target_arch = "x86_64")]
use sse2 as lanes;

/// `|x|` bits at or above which a lane takes the scalar code (`|x| ≥ 87.33655`, NaN, ±inf).
const SPECIAL: u32 = 0x42ae_ac50;
/// `|x|` bits above which `k = round(x / ln2)` (`|x| > 1.5 ln2`).
const BIG: u32 = 0x3f85_1592;
/// `|x|` bits above which `k = ±1` at least (`|x| > ½ ln2`).
const MID: u32 = 0x3eb1_7218;
/// `|x|` bits at or below which `expf(x) = 1 + x` (`|x| ≤ 2⁻¹⁴`).
const TINY: u32 = 0x3900_0000;

/// `x[i] = expf(x[i])` for every element.
pub(crate) fn expf_slice(x: &mut [f32]) {
    lanes::exp_in_place(x);
}

/// `x[i] = 1 / (1 + expf(-x[i]))` for every element: the logistic sigmoid.
pub(crate) fn sigmoidf_slice(x: &mut [f32]) {
    lanes::sigmoid_in_place(x);
}

/// `x[i] = x[i] * (1 / (1 + expf(-x[i])))` for every element: SiLU.
pub(crate) fn siluf_slice(x: &mut [f32]) {
    lanes::silu_in_place(x);
}

/// The scalar definitions, which the vector paths reproduce bit for bit: their tails, and the
/// whole slice on targets without a vector path.
#[cfg_attr(
    any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ),
    allow(dead_code)
)]
mod scalar {
    use super::super::expf;

    pub(super) fn sigmoid(v: f32) -> f32 {
        1.0 / (1.0 + expf(-v))
    }

    pub(super) fn exp_in_place(x: &mut [f32]) {
        for v in x {
            *v = expf(*v);
        }
    }

    pub(super) fn sigmoid_in_place(x: &mut [f32]) {
        for v in x {
            *v = sigmoid(*v);
        }
    }

    pub(super) fn silu_in_place(x: &mut [f32]) {
        for v in x {
            *v *= sigmoid(*v);
        }
    }
}

/// Recomputes with the scalar `expf` the lanes of `out` whose `special` mask lane is set
/// (non-zero), from the inputs `x`: the rare out-of-range lanes.
#[inline(never)]
#[cold]
fn fix_special(x: [f32; 4], special: [u32; 4], out: &mut [f32; 4]) {
    for i in 0..4 {
        if special[i] != 0 {
            out[i] = super::expf(x[i]);
        }
    }
}
