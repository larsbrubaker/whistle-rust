//! Tests for `expf_lanes`: this target's slice path (NEON, SSE2, simd128 or scalar) against the
//! scalar `expf`, `sigmoid` and `silu` bit for bit: a dense sweep of every 4096th f32, the
//! special values and both sides of every range boundary, at every lane position and in the
//! tails; and, ignored by default (seconds natively, too slow under wasm), all 2³² inputs.

use super::super::expf;
use super::{expf_slice, sigmoidf_slice, siluf_slice, BIG, MID, SPECIAL, TINY};

type SliceFn = fn(&mut [f32]);
type ScalarFn = fn(f32) -> f32;

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + expf(-v))
}

fn silu(v: f32) -> f32 {
    v * sigmoid(v)
}

const CASES: [(SliceFn, ScalarFn, &str); 3] = [
    (expf_slice, expf, "expf"),
    (sigmoidf_slice, sigmoid, "sigmoid"),
    (siluf_slice, silu, "silu"),
];

/// The same bits, or both NaN (a NaN's payload is not part of the contract: wasm leaves it
/// unspecified).
fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// The first input in `xs` where the slice path and the scalar function disagree.
fn first_mismatch(slice: SliceFn, scalar: ScalarFn, xs: &[f32]) -> Option<(f32, f32, f32)> {
    let mut got = xs.to_vec();
    slice(&mut got);
    xs.iter()
        .zip(&got)
        .map(|(&x, &g)| (x, g, scalar(x)))
        .find(|&(_, g, w)| !same(g, w))
}

/// Every 4096th bit pattern, the specials, and ±8 ulps around each range boundary, both signs.
fn gated_inputs() -> Vec<f32> {
    let mut bits: Vec<u32> = (0..1u32 << 20).map(|i| i << 12).collect();
    let edges = [
        0,
        TINY,
        MID,
        BIG,
        SPECIAL,
        0x42b1_7218, // overflow threshold, 88.72
        0x42cf_f1b5, // underflow-to-zero threshold, -103.97
        0x7f80_0000, // infinity
        0x7fc0_0000, // quiet NaN
    ];
    for e in edges {
        for d in 0..=16u32 {
            let b = (e + d).wrapping_sub(8) & 0x7fff_ffff;
            bits.extend([b, b | 0x8000_0000]);
        }
    }
    bits.extend([
        0x0000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0x7f80_0001,
    ]);
    bits.into_iter().map(f32::from_bits).collect()
}

#[test]
fn slices_match_the_scalar_bits_over_the_gated_sweep() {
    let xs = gated_inputs();
    for (slice, scalar, name) in CASES {
        // Offsets 0..4 put every input in every lane, and lengths 0..=7 cover every tail.
        for skip in 0..4 {
            if let Some((x, g, w)) = first_mismatch(slice, scalar, &xs[skip..]) {
                panic!(
                    "{name}({x:e} = {:#010x}): slice {:#010x}, scalar {:#010x}",
                    x.to_bits(),
                    g.to_bits(),
                    w.to_bits()
                );
            }
        }
        for len in 0..=7 {
            let short = &xs[xs.len() - 64..xs.len() - 64 + len];
            assert!(
                first_mismatch(slice, scalar, short).is_none(),
                "{name} len {len}"
            );
        }
    }
}

#[test]
fn slices_match_the_scalar_bits_on_model_ranges() {
    // Activations and softmax arguments: dense around zero, where the k = 0 and k = ±1
    // branches live, with every input in every lane.
    let xs: Vec<f32> = (-200_000..200_000).map(|i| i as f32 * 1e-4).collect();
    for (slice, scalar, name) in CASES {
        for skip in 0..4 {
            assert!(
                first_mismatch(slice, scalar, &xs[skip..]).is_none(),
                "{name}"
            );
        }
    }
}

#[test]
#[ignore = "all 2^32 inputs: a few seconds natively in release, far longer in debug or wasm"]
fn slices_match_the_scalar_bits_on_every_f32() {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    for (slice, scalar, name) in CASES {
        let next = std::sync::atomic::AtomicU32::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut xs = vec![0.0f32; 1 << 16];
                    loop {
                        let block = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if block >= 1 << 16 {
                            break;
                        }
                        for (i, x) in xs.iter_mut().enumerate() {
                            *x = f32::from_bits(block << 16 | i as u32);
                        }
                        if let Some((x, g, w)) = first_mismatch(slice, scalar, &xs) {
                            panic!(
                                "{name}({:#010x}): slice {:#010x}, scalar {:#010x}",
                                x.to_bits(),
                                g.to_bits(),
                                w.to_bits()
                            );
                        }
                    }
                });
            }
        });
    }
}
