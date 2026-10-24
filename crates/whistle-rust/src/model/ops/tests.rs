// Unit tests for the model primitives in `ops`: A8 on hand-computed rows (ties, clamping, an
// all-zero row), Sinkhorn's doubly stochastic output, RoPE's tables and rotation against
// hand-evaluated values, the small activations, and softmax over zero-wide rows.

use super::*;

#[test]
fn a8_rounds_to_steps_of_absmax_over_127() {
    // absmax 127 gives scale 1: values round half to even onto integers.
    let mut x = vec![127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.4];
    a8(&mut x, 8);
    assert_eq!(x, vec![127.0, 0.0, 2.0, 2.0, -0.0, -2.0, -2.0, 3.0]);
}

#[test]
fn a8_scales_each_row_by_its_own_absmax() {
    // Row 1: absmax 254, scale 2; 3 / 2 = 1.5 rounds to 2 (even) -> 4; 5 / 2 = 2.5 -> 2 -> 4.
    // Row 2: absmax 12.7, scale 0.1 (in f32); 0.05 / 0.1 = 0.5 rounds to 0.
    let mut x = vec![254.0, 3.0, 5.0, -254.0, 12.7, 0.05, -12.7, 1.0];
    a8(&mut x, 4);
    assert_eq!(&x[..4], &[254.0, 4.0, 4.0, -254.0]);
    let scale = 12.7f32 / 127.0;
    let expect = |v: f32| {
        let q = (v / scale).round_ties_even() * scale;
        v + (q - v)
    };
    assert_eq!(x[4], expect(12.7));
    assert_eq!(x[5], 0.0);
    assert_eq!(x[6], expect(-12.7));
    assert_eq!(x[7], expect(1.0));
}

#[test]
fn a8_leaves_an_all_zero_row_zero() {
    let mut x = vec![0.0f32; 6];
    a8(&mut x, 6);
    assert!(x.iter().all(|&v| v == 0.0));
}

#[test]
fn fake_quant_uses_the_row_absmax_at_other_widths() {
    // 4 bits: qmax 7. absmax 7 gives scale 1, so values round half to even onto integers, and
    // a row whose most negative value is the absmax maps it to -7 (the extra level -8 is never
    // reached by symmetric scaling).
    let mut x = vec![-7.0, 6.6, -6.5, 0.4];
    fake_quant_rows(&mut x, 4, 4);
    assert_eq!(x, vec![-7.0, 7.0, -6.0, 0.0]);
}

#[test]
fn sinkhorn_rows_and_columns_sum_to_one() {
    // Logits shaped like the mHC residual mix: `b_res` starts at 4·I, plus a learned spread.
    // The last half-step normalises columns, so they sum to 1 to f32 rounding; rows carry what
    // 20 rounds leave unconverged, which the same algorithm in float64 (numpy) puts at these
    // sums.
    let row_sums = [1.000_137_7f32, 0.999_925_1, 0.999_749_7, 1.000_187_5];
    let mut m: Vec<f32> = (0..16).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3).collect();
    for d in 0..4 {
        m[d * 5] += 4.0;
    }
    sinkhorn(&mut m, 4);
    for r in 0..4 {
        let row: f32 = m[r * 4..r * 4 + 4].iter().sum();
        assert!((row - row_sums[r]).abs() < 2e-6, "row {r} sums to {row}");
    }
    for c in 0..4 {
        let col: f32 = (0..4).map(|r| m[r * 4 + c]).sum();
        assert!((col - 1.0).abs() < 1e-6, "column {c} sums to {col}");
    }
    assert!(m.iter().all(|&v| v > 0.0));
}

#[test]
fn rope_tables_follow_the_frequency_formula() {
    // head_dim 4, theta 100: freqs 1 and 100^(-1/2) = 0.1.
    let rope = Rope::new(4, 3, 100.0);
    assert_eq!(rope.half, 2);
    assert_eq!(&rope.cos[..2], &[1.0, 1.0]);
    assert_eq!(&rope.sin[..2], &[0.0, 0.0]);
    let (c, s) = (&rope.cos[4..6], &rope.sin[4..6]);
    assert!((c[0] - 2.0f32.cos()).abs() < 1e-6 && (s[0] - 2.0f32.sin()).abs() < 1e-6);
    assert!((c[1] - 0.2f32.cos()).abs() < 1e-6 && (s[1] - 0.2f32.sin()).abs() < 1e-6);
}

#[test]
fn rope_rotates_the_two_halves_as_pairs() {
    let rope = Rope::new(4, 2, 100.0);
    // Position 1: angles 1 and 0.1. x = [x1 | x2] = [1, 2 | 3, 4].
    let mut x = [1.0f32, 2.0, 3.0, 4.0];
    rope.apply(&mut x, 1);
    let (c0, s0, c1, s1) = (1.0f32.cos(), 1.0f32.sin(), 0.1f32.cos(), 0.1f32.sin());
    let want = [
        1.0 * c0 - 3.0 * s0,
        2.0 * c1 - 4.0 * s1,
        3.0 * c0 + 1.0 * s0,
        4.0 * c1 + 2.0 * s1,
    ];
    for (g, w) in x.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "{x:?} against {want:?}");
    }
    // Position 0 is the identity.
    let mut y = [1.0f32, 2.0, 3.0, 4.0];
    rope.apply(&mut y, 0);
    assert_eq!(y, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn activations_and_norms_match_their_definitions() {
    assert_eq!(sigmoid(0.0), 0.5);
    let mut s = [0.0f32, 2.0];
    silu_in_place(&mut s);
    assert_eq!(s[0], 0.0);
    assert!((s[1] - 2.0 / (1.0 + (-2.0f32).exp())).abs() < 1e-7);
    let g = glu(&[1.0, 2.0, 0.0, 3.0], 2);
    assert_eq!(g, vec![0.5, 2.0 * sigmoid(3.0)]);
    let mut p = vec![1.0f32, 1.0, 1.0, 1.0];
    softmax(&mut p, 4);
    assert_eq!(p, vec![0.25; 4]);
    // mean(x²) = (9 + 16) / 2 = 12.5.
    let n = zc_rms_norm(&[3.0, 4.0], &[0.0, 1.0]);
    let rms = (12.5f32 + 1e-6).sqrt();
    assert_eq!(n, vec![3.0 / rms, 2.0 * 4.0 / rms]);
    let u = rms_unit(&[3.0, 4.0], 2);
    assert_eq!(u, vec![3.0 * (1.0 / rms), 4.0 * (1.0 / rms)]);
    assert_eq!(
        linear(&[1.0, 2.0], &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 3),
        vec![11.0, 17.0, 23.0]
    );
}

#[test]
fn softmax_over_zero_wide_rows_is_a_no_op() {
    let mut empty: Vec<f32> = Vec::new();
    softmax(&mut empty, 0);
    assert!(empty.is_empty());
}
