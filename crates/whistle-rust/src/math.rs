//! Deterministic transcendentals: `exp`, `ln`, `log10`, `sin`, `cos` and `pow` in f32 and f64,
//! computed in plain IEEE arithmetic so every target (aarch64, x86_64, wasm32 with or without
//! simd128) produces the same bits for the same input.
//!
//! `std`'s `f32::exp` and friends call the platform libm (Apple's, glibc, MSVC, or Rust's
//! bundled one on wasm), and those round differently in the last bit. One flipped bit in the
//! stem's `silu` crosses an A8 rounding step and changes the decode, so the library computes
//! every transcendental here instead. The routines are the Rust `libm` crate's ports of musl
//! libm (MIT; the algorithms are FreeBSD msun / Sun fdlibm), at their accuracy: under 1 ulp,
//! usually correctly rounded. They use only `+ - * /`, integer bit manipulation, `floor` and
//! `sqrt` (both exact in IEEE 754); Rust never contracts `a * b + c` into a fused multiply-add,
//! so the compiler cannot change the rounding either. `scripts/wasm-test.sh` and the
//! `identity` test (`tests/transcribe/identity.rs`) check the result end to end. The `deterministic_math` gate keeps std
//! transcendentals out of the library.

// The ports keep the source's structure and its published constants digit for digit, so a
// reader can check them against it: `x - x` deliberately makes NaN from inf, the constants carry
// more digits than f64 holds, and the loops index tables as the original does.
#![allow(
    clippy::approx_constant,
    clippy::eq_op,
    clippy::excessive_precision,
    clippy::explicit_counter_loop,
    clippy::if_same_then_else,
    clippy::needless_else,
    clippy::needless_late_init,
    clippy::needless_range_loop
)]

mod cos;
mod cosf;
mod exp;
mod expf;
mod k_cos;
mod k_cosf;
mod k_sin;
mod k_sinf;
mod log;
mod log10;
mod logf;
mod pow;
mod powf;
mod rem_pio2;
mod rem_pio2_large;
mod rem_pio2f;
mod sin;
mod sinf;
#[cfg(test)]
mod tests;

use k_cos::k_cos;
use k_cosf::k_cosf;
use k_sin::k_sin;
use k_sinf::k_sinf;
use rem_pio2::rem_pio2;
use rem_pio2_large::rem_pio2_large;
use rem_pio2f::rem_pio2f;

pub(crate) use cos::cos;
pub(crate) use cosf::cosf;
pub(crate) use exp::exp;
pub(crate) use expf::expf;
pub(crate) use log::log as ln;
pub(crate) use log10::log10;
pub(crate) use logf::logf as lnf;
pub(crate) use pow::pow;
pub(crate) use powf::powf;
pub(crate) use sin::sin;
pub(crate) use sinf::sinf;

/// `x * 2^n` (musl `scalbnf`), computed so intermediate scalings never overflow or flush early.
fn scalbnf(x: f32, mut n: i32) -> f32 {
    let x1p127 = f32::from_bits(0x7f00_0000); // 2^127
    let x1p_126 = f32::from_bits(0x0080_0000); // 2^-126
    let x1p24 = f32::from_bits(0x4b80_0000); // 2^24
    let mut y = x;
    if n > 127 {
        y *= x1p127;
        n -= 127;
        if n > 127 {
            y *= x1p127;
            n -= 127;
            if n > 127 {
                n = 127;
            }
        }
    } else if n < -126 {
        // Scale by 2^-102 in two steps so a subnormal result rounds once, not twice.
        y *= x1p_126 * x1p24;
        n += 126 - 24;
        if n < -126 {
            y *= x1p_126 * x1p24;
            n += 126 - 24;
            if n < -126 {
                n = -126;
            }
        }
    }
    // n is now in [-126, 127], so the biased exponent is in [1, 254].
    y * f32::from_bits(((0x7f + n) as u32) << 23)
}

/// `x * 2^n` (musl `scalbn`), f64.
fn scalbn(x: f64, mut n: i32) -> f64 {
    let x1p1023 = f64::from_bits(0x7fe0_0000_0000_0000); // 2^1023
    let x1p53 = f64::from_bits(0x4340_0000_0000_0000); // 2^53
    let x1p_1022 = f64::from_bits(0x0010_0000_0000_0000); // 2^-1022
    let mut y = x;
    if n > 1023 {
        y *= x1p1023;
        n -= 1023;
        if n > 1023 {
            y *= x1p1023;
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        y *= x1p_1022 * x1p53;
        n += 1022 - 53;
        if n < -1022 {
            y *= x1p_1022 * x1p53;
            n += 1022 - 53;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    // n is now in [-1022, 1023], so the biased exponent is in [1, 2046].
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

fn fabsf(x: f32) -> f32 {
    x.abs()
}

fn fabs(x: f64) -> f64 {
    x.abs()
}

/// IEEE 754 square root is correctly rounded on every target, so std's is deterministic.
fn sqrtf(x: f32) -> f32 {
    x.sqrt()
}

fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

fn get_high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

fn with_set_high_word(f: f64, hi: u32) -> f64 {
    let mut tmp = f.to_bits();
    tmp &= 0x0000_0000_ffff_ffff;
    tmp |= (hi as u64) << 32;
    f64::from_bits(tmp)
}

fn with_set_low_word(f: f64, lo: u32) -> f64 {
    let mut tmp = f.to_bits();
    tmp &= 0xffff_ffff_0000_0000;
    tmp |= lo as u64;
    f64::from_bits(tmp)
}
