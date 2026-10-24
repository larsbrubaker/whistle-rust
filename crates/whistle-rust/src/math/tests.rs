//! Tests for `math`: ulp error against higher-precision references over dense sweeps, the
//! special values, the ranges the model feeds each function, and known-answer hashes of the
//! exact output bits, so a change of a single bit on any target fails here.

use super::*;

mod double_double;

/// Distance in units in the last place between two finite f32 values of the same sign class.
fn ulps_f32(a: f32, b: f32) -> u64 {
    let key = |x: f32| {
        let bits = x.to_bits() as i64;
        if bits < 0x8000_0000 {
            bits
        } else {
            0x8000_0000 - bits
        }
    };
    (key(a) - key(b)).unsigned_abs()
}

fn ulps_f64(a: f64, b: f64) -> u64 {
    let key = |x: f64| {
        let bits = x.to_bits() as i128;
        if bits < 0x8000_0000_0000_0000 {
            bits
        } else {
            0x8000_0000_0000_0000 - bits
        }
    };
    (key(a) - key(b)).unsigned_abs() as u64
}

/// About `count` f32 bit patterns evenly spaced in `[lo, hi]` (both positive finite).
fn sweep_f32(lo: f32, hi: f32, count: u32) -> impl Iterator<Item = f32> {
    let (a, b) = (lo.to_bits(), hi.to_bits());
    let step = ((b - a) / count).max(1);
    (a..=b).step_by(step as usize).map(f32::from_bits)
}

fn sweep_f64(lo: f64, hi: f64, count: u64) -> impl Iterator<Item = f64> {
    let (a, b) = (lo.to_bits(), hi.to_bits());
    // Index by count, not `step_by` over the u64 range, whose length overflows usize on wasm32.
    let step = ((b - a) / count).max(1);
    (0..=(b - a) / step).map(move |i| f64::from_bits(a + i * step))
}

/// The worst ulp error of an f32 function against its f64 reference over `inputs`.
fn worst_f32(f: fn(f32) -> f32, reference: fn(f64) -> f64, inputs: &[f32]) -> (u64, f32) {
    let mut worst = (0, 0.0);
    for &x in inputs {
        let want = reference(f64::from(x)) as f32;
        let e = ulps_f32(f(x), want);
        if e > worst.0 {
            worst = (e, x);
        }
    }
    worst
}

fn both_signs(v: Vec<f32>) -> Vec<f32> {
    v.iter().flat_map(|&x| [x, -x]).collect()
}

#[test]
fn f32_functions_are_within_one_ulp_of_the_f64_reference() {
    let exp_in = both_signs(sweep_f32(1e-30, 87.0, 400_000).collect());
    assert!(
        worst_f32(expf, f64::exp, &exp_in).0 <= 1,
        "expf {:?}",
        worst_f32(expf, f64::exp, &exp_in)
    );
    let log_in: Vec<f32> = sweep_f32(f32::from_bits(1), f32::MAX, 400_000).collect();
    assert!(
        worst_f32(lnf, f64::ln, &log_in).0 <= 1,
        "logf {:?}",
        worst_f32(lnf, f64::ln, &log_in)
    );
    // Small, medium (Cody-Waite) and huge (Payne-Hanek) arguments.
    let trig_in = both_signs(sweep_f32(1e-30, 1e30, 400_000).collect());
    assert!(
        worst_f32(sinf, f64::sin, &trig_in).0 <= 1,
        "sinf {:?}",
        worst_f32(sinf, f64::sin, &trig_in)
    );
    assert!(
        worst_f32(cosf, f64::cos, &trig_in).0 <= 1,
        "cosf {:?}",
        worst_f32(cosf, f64::cos, &trig_in)
    );
    let mut worst = 0;
    for x in sweep_f32(1e-3, 1e4, 600) {
        for y in [-3.7f32, -1.0, -0.5, 0.25, 0.5, 1.0 / 3.0, 2.0, 7.5] {
            let want = f64::from(x).powf(f64::from(y));
            if want.is_finite() && want > 1e-37 && want < 1e38 {
                worst = worst.max(ulps_f32(powf(x, y), want as f32));
            }
        }
    }
    assert!(worst <= 1, "powf {worst}");
}

/// The f64 functions against the truth, not against the platform libm: glibc, Apple's libm and
/// this port can each be up to about an ulp from the true value, on opposite sides, so two
/// correct implementations may differ by 2 ulps and std is no oracle. `exp`, `ln`, `log10` and
/// `pow` are measured against double-double references (`double_double.rs`), rounded to f64:
/// the port is within 1 ulp of the correctly rounded value. `sin` and `cos` have no such
/// reference here, so they are compared with std within 2 ulps (each side within 1 of the truth).
#[test]
fn f64_functions_are_within_one_ulp_of_the_true_value() {
    type Pair = (fn(f64) -> f64, fn(f64) -> f64, &'static str, u64);
    let pairs: [Pair; 5] = [
        (exp, |x| double_double::exp(x).hi, "exp", 1),
        (ln, |x| double_double::ln(x).hi, "log", 1),
        (log10, |x| double_double::log10(x).hi, "log10", 1),
        (sin, f64::sin, "sin", 2),
        (cos, f64::cos, "cos", 2),
    ];
    for (f, reference, name, budget) in pairs {
        let hi = if name == "exp" { 709.0 } else { 1e300 };
        let lo = if name == "exp" {
            1e-300
        } else {
            f64::from_bits(1)
        };
        let mut inputs: Vec<f64> = sweep_f64(lo, hi, 200_000).collect();
        if name == "exp" || name == "sin" || name == "cos" {
            inputs.extend(inputs.clone().iter().map(|x| -x));
        }
        for x in inputs {
            let (got, want) = (f(x), reference(x));
            assert!(
                ulps_f64(got, want) <= budget,
                "{name}({x:e}) = {got:e}, reference {want:e}"
            );
        }
    }
    let pow_ref = |x: f64, y: f64| double_double::pow(x, y).hi;
    for x in sweep_f64(1e-3, 1e4, 2000) {
        for y in [-3.7, -1.0, -0.5, 0.25, 1.0 / 3.0, 2.0, 7.5] {
            assert!(ulps_f64(pow(x, y), pow_ref(x, y)) <= 1, "pow({x:e}, {y})");
        }
    }
    for db in -120..=40 {
        let y = f64::from(db) / 10.0;
        assert!(
            ulps_f64(pow(10.0, y), pow_ref(10.0, y)) <= 1,
            "pow(10, {y})"
        );
    }
}

#[test]
fn special_values_follow_ieee() {
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    assert!(expf(nan).is_nan() && lnf(nan).is_nan() && sinf(nan).is_nan() && cosf(nan).is_nan());
    assert_eq!(expf(inf), inf);
    assert_eq!(expf(-inf), 0.0);
    assert_eq!(expf(0.0), 1.0);
    assert_eq!(expf(-0.0), 1.0);
    assert_eq!(expf(88.8), inf, "overflow");
    assert_eq!(expf(-104.0), 0.0, "underflow");
    assert!(
        expf(-100.0) > 0.0 && expf(-100.0) < f32::MIN_POSITIVE,
        "subnormal result"
    );
    assert_eq!(lnf(0.0), -inf);
    assert_eq!(lnf(-0.0), -inf);
    assert!(lnf(-1.0).is_nan());
    assert_eq!(lnf(inf), inf);
    assert_eq!(lnf(1.0), 0.0);
    assert!(
        (lnf(f32::from_bits(1)) - (-103.27893)).abs() < 1e-4,
        "subnormal input"
    );
    assert!(sinf(inf).is_nan() && cosf(-inf).is_nan());
    assert_eq!(sinf(-0.0).to_bits(), (-0.0f32).to_bits());
    assert_eq!(cosf(0.0), 1.0);
    assert_eq!(powf(0.0, 0.0), 1.0);
    assert_eq!(powf(2.0, 0.5), std::f32::consts::SQRT_2);
    assert!(powf(-2.0, 0.5).is_nan());
    assert_eq!(powf(inf, -1.0), 0.0);
    assert_eq!(exp(f64::INFINITY), f64::INFINITY);
    assert_eq!(exp(-f64::INFINITY), 0.0);
    assert_eq!(exp(710.0), f64::INFINITY);
    assert!(exp(-740.0) > 0.0 && exp(-746.0) == 0.0);
    assert_eq!(ln(0.0), -f64::INFINITY);
    assert!(ln(-1.0).is_nan() && log10(-1.0).is_nan() && sin(f64::NAN).is_nan());
    assert_eq!(log10(1000.0), 3.0);
    assert_eq!(log10(0.0), -f64::INFINITY);
    assert_eq!(pow(10.0, 2.0), 100.0);
    assert_eq!(pow(0.0, -1.0), f64::INFINITY);
    assert_eq!(cos(0.0), 1.0);
    assert_eq!(sin(-0.0).to_bits(), (-0.0f64).to_bits());
}

#[test]
fn model_ranges_are_accurate() {
    // sigmoid / silu: expf(-x) over the activations' range, softmax: expf(v - max) <= 1.
    let act: Vec<f32> = (-200_000..=200_000).map(|i| i as f32 * 1e-4).collect();
    assert!(worst_f32(expf, f64::exp, &act).0 <= 1);
    let soft: Vec<f32> = (0..200_000).map(|i| -(i as f32) * 5e-4).collect();
    assert!(worst_f32(expf, f64::exp, &soft).0 <= 1);
    // RoPE angles: position (up to a few thousand) times the head frequencies.
    let mut angles = Vec::new();
    for t in 0..4096 {
        for i in 0..32 {
            let f = 1.0 / powf(10000.0, (2 * i) as f32 / 64.0);
            angles.push(t as f32 * f);
        }
    }
    assert!(worst_f32(sinf, f64::sin, &angles).0 <= 1);
    assert!(worst_f32(cosf, f64::cos, &angles).0 <= 1);
}

/// FNV-1a over the little-endian bytes of each output's bits (byte-wise; `examples/stage_hashes.rs`
/// and the identity test fold whole words instead, so the two hashes are not comparable).
fn fnv(bits: impl Iterator<Item = u64>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bits {
        for byte in b.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// 400 inputs spread over each function's interesting range, deterministic.
fn inputs_f32(lo: f32, hi: f32) -> impl Iterator<Item = f32> {
    (0..400).map(move |i| lo + (hi - lo) * (i as f32 / 399.0))
}

#[test]
fn known_answers_are_bit_exact() {
    let f32_hash = |f: fn(f32) -> f32, lo: f32, hi: f32| {
        fnv(inputs_f32(lo, hi).map(|x| u64::from(f(x).to_bits())))
    };
    let f64_hash = |f: fn(f64) -> f64, lo: f64, hi: f64| {
        fnv((0..400).map(|i| f(lo + (hi - lo) * (f64::from(i) / 399.0)).to_bits()))
    };
    let got = [
        f32_hash(expf, -90.0, 90.0),
        f32_hash(lnf, 1e-6, 1e6),
        f32_hash(sinf, -1e4, 1e4),
        f32_hash(cosf, -1e4, 1e4),
        f32_hash(|x| powf(10000.0, x), -1.0, 1.0),
        f64_hash(exp, -700.0, 700.0),
        f64_hash(ln, 1e-12, 1e12),
        f64_hash(log10, 1e-12, 1e12),
        f64_hash(sin, -1e6, 1e6),
        f64_hash(cos, -1e6, 1e6),
        f64_hash(|y| pow(10.0, y), -12.0, 4.0),
    ];
    assert_eq!(got, KNOWN, "a transcendental changed its output bits");
    // A few exact values, readable without the hash.
    assert_eq!(expf(1.0).to_bits(), 0x402d_f854);
    assert_eq!(lnf(10.0).to_bits(), 0x4013_5d8e);
    assert_eq!(sinf(1.0).to_bits(), 0x3f57_6aa4);
    assert_eq!(cosf(1.0).to_bits(), 0x3f0a_5140);
    // The argument reductions' other paths: f32 Cody-Waite (|x| < 2^28 pi/2) at +-1e4, Payne-Hanek
    // (`rem_pio2_large`) beyond it; f64 medium (|x| < 2^20 pi/2) at +-1e6, large beyond. sin(1e22)
    // is the classic hard case, -0.8522008497671888.
    let f32_trig = [
        (1e4f32, 0xbe9c_797d, 0xbf73_c074),
        (-1e4, 0x3e9c_797d, 0xbf73_c074),
        (1e9, 0x3f0b_bc65, 0x3f56_7fc6),
        (-3e38, 0xbf5f_f9c4, 0xbef7_f57c),
    ];
    for (x, s, c) in f32_trig {
        assert_eq!(
            (sinf(x).to_bits(), cosf(x).to_bits()),
            (s, c),
            "sinf/cosf({x:e})"
        );
    }
    let f64_trig = [
        (1e6f64, 0xbfd6_664b_2568_d867, 0x3fed_f9df_9906_d32c),
        (-1e6, 0x3fd6_664b_2568_d867, 0x3fed_f9df_9906_d32c),
        (1e7, 0x3fda_ea41_4a8a_3352, 0xbfed_085b_e7a8_f4a1),
        (1e22, 0xbfeb_453a_b76b_f397, 0x3fe0_be2c_ef01_c8f4),
        (1e300, 0xbfea_2c16_b010_e385, 0xbfe2_6990_22ad_c4c1),
    ];
    for (x, s, c) in f64_trig {
        assert_eq!(
            (sin(x).to_bits(), cos(x).to_bits()),
            (s, c),
            "sin/cos({x:e})"
        );
    }
}

/// Measured on aarch64 macOS; `scripts/wasm-test.sh` checks the same values on wasm32.
const KNOWN: [u64; 11] = [
    0xce6193c0c5624b04,
    0x6b08340cc428f897,
    0x619d03ad7df99876,
    0xec603bfbce6843df,
    0x87583d2ecc4cfa4e,
    0xdbc791465835650e,
    0x0915ca8840053f13,
    0x00562ac73a847637,
    0x3593220d1892334e,
    0x59324bcbd4ff12ad,
    0x5b74d5aecbab76cb,
];
