// Unit tests for the dense kernel in `kernels`: the scalar and SIMD paths against the port's
// sequential sum (bit-identical, since they keep its order), on shapes that are not multiples
// of the panel or row-block sizes, signed zeros, and bit-identical projections for every thread
// count.

use super::*;

/// Deterministic values in [-1, 1) from a small LCG.
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

/// The sequential sum the port has always used, per output.
fn sequential(x: &[f32], w: &[f32], n: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for xr in x.chunks_exact(n) {
        for wr in w.chunks_exact(n) {
            out.push(xr.iter().zip(wr).map(|(a, b)| a * b).sum());
        }
    }
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

const SHAPES: [(usize, usize, usize); 8] = [
    (1, 1, 1),
    (1, 7, 3),
    (2, 8, 4),
    (3, 9, 5),
    (5, 17, 13),
    (7, 64, 9),
    (1, 515, 33),
    (9, 130, 1030),
];

#[test]
fn the_fastest_path_matches_the_scalar_path_bit_for_bit() {
    for &(rows, n, outputs) in &SHAPES {
        let (x, w) = (values(rows * n, 1), values(outputs * n, 2));
        let mut scalar = vec![0.0; rows * outputs];
        let mut fast = vec![0.0; rows * outputs];
        linear_into(Path::Scalar, &x, &w, n, &mut scalar);
        linear_into(best_path(), &x, &w, n, &mut fast);
        assert_eq!(
            scalar,
            fast,
            "shape {rows}x{n}x{outputs} on {:?}",
            best_path()
        );
    }
}

#[test]
fn every_path_reproduces_the_sequential_sum_bit_for_bit() {
    for &(rows, n, outputs) in &SHAPES {
        let (x, w) = (values(rows * n, 3), values(outputs * n, 4));
        let want = bits(&sequential(&x, &w, n));
        assert_eq!(
            bits(&linear_with(Path::Scalar, 1, &x, &w, n, outputs)),
            want
        );
        assert_eq!(bits(&linear_with(best_path(), 1, &x, &w, n, outputs)), want);
    }
}

#[test]
fn the_few_row_path_reproduces_the_sequential_sum_on_every_tail_shape() {
    // 1–3 rows take the in-register transpose (`small`): inputs past a multiple of 4 and
    // outputs past the last group of 4 finish by the scalar sum, in the same order.
    for rows in 1..=3 {
        for n in [1, 2, 3, 4, 5, 7, 8, 17, 64, 130] {
            for outputs in [1, 3, 4, 5, 15, 16, 17, 20, 33, 66] {
                let (x, w) = (values(rows * n, 13), values(outputs * n, 14));
                let want = bits(&sequential(&x, &w, n));
                for path in [Path::Scalar, best_path()] {
                    let mut out = vec![f32::NAN; rows * outputs];
                    linear_into(path, &x, &w, n, &mut out);
                    assert_eq!(bits(&out), want, "{rows}x{n}x{outputs} on {path:?}");
                }
            }
        }
    }
}

#[test]
fn an_all_negative_zero_product_keeps_its_sign_as_the_sequential_sum_does() {
    let (x, w) = (vec![-0.0f32, 0.0, -1.0], vec![1.0f32, -2.0, 0.0]);
    let want = bits(&sequential(&x, &w, 3));
    assert_eq!(bits(&linear_with(Path::Scalar, 1, &x, &w, 3, 1)), want);
    assert_eq!(bits(&linear_with(best_path(), 1, &x, &w, 3, 1)), want);
}

#[test]
fn every_thread_count_gives_bit_identical_projections() {
    // Row counts that split evenly, unevenly, barely, and not at all (one and three rows stay
    // on one thread), all above the work threshold.
    for &(rows, n, outputs) in &[
        (64, 256, 128),
        (100, 300, 70),
        (17, 1024, 777),
        (1, 512, 4099),
        (3, 1024, 777),
    ] {
        let (x, w) = (values(rows * n, 7), values(outputs * n, 8));
        let one = linear_with(best_path(), 1, &x, &w, n, outputs);
        for t in [2, 3, 4, 7, usize::MAX] {
            assert_eq!(
                one,
                linear_with(best_path(), t, &x, &w, n, outputs),
                "{t} threads"
            );
        }
    }
}

#[test]
fn signed_zero_products_through_the_vector_part_keep_the_sequential_sums_sign() {
    // Every product is a signed zero, over 8+ inputs and 4+ outputs, so the vector parts (not
    // only the scalar tails) decide each sign: an output whose products are all -0.0 must stay
    // -0.0, as the sequential sum from -0.0 does, and one +0.0 product turns it to +0.0. Inputs
    // alternate +0.0 and -0.0 against weights alternating -1.5 and 2.0 (all -0.0 products when
    // the parities line up, +0.0 ones when an odd width shifts them); output 1 starts with a
    // +0.0 product. Both signs are asserted present.
    for &(rows, n, outputs) in &[(1, 8, 4), (2, 9, 5), (3, 16, 17), (4, 12, 16), (5, 33, 20)] {
        let x: Vec<f32> = (0..rows * n)
            .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
            .collect();
        let mut w: Vec<f32> = (0..outputs * n)
            .map(|i| if i % 2 == 0 { -1.5 } else { 2.0 })
            .collect();
        // Output 1's first weight makes its first product +0.0 · +1.0 = +0.0.
        w[n] = 1.0;
        let want = bits(&sequential(&x, &w, n));
        assert!(want.contains(&(-0.0f32).to_bits()) && want.contains(&0.0f32.to_bits()));
        for path in [Path::Scalar, best_path()] {
            let mut out = vec![f32::NAN; rows * outputs];
            linear_into(path, &x, &w, n, &mut out);
            assert_eq!(bits(&out), want, "{rows}x{n}x{outputs} on {path:?}");
        }
    }
}

#[test]
fn zero_outputs_project_to_nothing_on_every_row_count_and_width() {
    for rows in [0, 1, 3, 4, 9] {
        for n in [0, 1, 5] {
            let x = values(rows * n, 19);
            for path in [Path::Scalar, best_path()] {
                let mut out: Vec<f32> = Vec::new();
                linear_into(path, &x, &[], n, &mut out);
                assert!(out.is_empty(), "{rows}x{n}x0 on {path:?}");
                assert!(linear_with(path, 1, &x, &[], n, 0).is_empty());
            }
        }
    }
}

#[test]
fn an_empty_input_width_projects_to_zeros() {
    assert_eq!(
        linear_with(Path::Scalar, 1, &[], &[], 0, 3),
        Vec::<f32>::new()
    );
}

#[test]
fn a_huge_thread_setting_is_capped_and_projects_without_overflow() {
    with_threads(usize::MAX, || assert_eq!(threads(), MAX_THREADS));
    set_threads(usize::MAX);
    assert_eq!(threads(), MAX_THREADS);
    set_threads(0);
    let (x, w) = (values(512 * 64, 9), values(64 * 64, 10));
    assert_eq!(
        linear_with(best_path(), usize::MAX, &x, &w, 64, 64),
        linear_with(best_path(), 1, &x, &w, 64, 64)
    );
}

/// The measurement behind the few-row path: best of 50 single-row projections of the decoder
/// step's shapes, by transposed panels and by the in-register transpose. Run in release:
/// `cargo test --release -p whistle-rust --lib few_row_timing -- --ignored --nocapture`.
#[test]
#[ignore = "release measurement for the few-row path"]
fn few_row_timing() {
    for (n, outputs) in [(512, 8199), (512, 512), (512, 1024), (1024, 512)] {
        let (x, w) = (values(n, 15), values(outputs * n, 16));
        let mut out = vec![0.0f32; outputs];
        let mut best = |f: &mut dyn FnMut(&mut [f32])| {
            (0..50)
                .map(|_| {
                    let start = std::time::Instant::now();
                    f(&mut out);
                    start.elapsed().as_secs_f64() * 1e6
                })
                .fold(f64::INFINITY, f64::min)
        };
        let panels = best(&mut |o| linear_panels_into(best_path(), &x, &w, n, o, w.len() / n, 0));
        let small = best(&mut |o| small::linear_rows_into(best_path(), &x, &w, n, o));
        println!("1x{n}x{outputs}: panels {panels:.0} us, in-register {small:.0} us");
    }
    // A decoder step streams about 72 MB of weights per token, more than the caches hold:
    // 64 distinct 512x512 matrices in turn, best of 10 passes.
    let (n, outputs, count) = (512, 512, 64);
    let (x, w) = (values(n, 17), values(count * outputs * n, 18));
    let mut out = vec![0.0f32; outputs];
    let mut best = |f: &mut dyn FnMut(&[f32], &mut [f32])| {
        (0..10)
            .map(|_| {
                let start = std::time::Instant::now();
                for m in w.chunks_exact(outputs * n) {
                    f(m, &mut out);
                }
                start.elapsed().as_secs_f64() * 1e6
            })
            .fold(f64::INFINITY, f64::min)
    };
    let panels = best(&mut |m, o| linear_panels_into(best_path(), &x, m, n, o, m.len() / n, 0));
    let small = best(&mut |m, o| small::linear_rows_into(best_path(), &x, m, n, o));
    println!("{count} x 1x{n}x{outputs} (64 MB): panels {panels:.0} us, in-register {small:.0} us");
}

/// The measurement behind `PARALLEL_MIN_MACS`: best of 30 runs of a projection on 1 and 4
/// threads around the threshold. Run in release:
/// `cargo test --release -p whistle-rust --lib parallel_threshold -- --ignored --nocapture`.
#[test]
#[ignore = "release measurement for PARALLEL_MIN_MACS"]
fn parallel_threshold_timing() {
    for outputs in [16, 32, 64, 128, 256, 512] {
        let (rows, n) = (64, 256);
        let (x, w) = (values(rows * n, 11), values(outputs * n, 12));
        let best = |t: usize| {
            (0..30)
                .map(|_| {
                    let start = std::time::Instant::now();
                    // Below the threshold `linear_with` would not split, so split by hand.
                    let per = rows.div_ceil(t);
                    let mut out = vec![0.0f32; rows * outputs];
                    std::thread::scope(|s| {
                        for (xc, oc) in x.chunks(per * n).zip(out.chunks_mut(per * outputs)) {
                            let w = &w;
                            s.spawn(move || linear_into(best_path(), xc, w, n, oc));
                        }
                    });
                    start.elapsed().as_secs_f64() * 1e6
                })
                .fold(f64::INFINITY, f64::min)
        };
        println!(
            "{} MACs: 1 thread {:.0} us, 4 threads {:.0} us",
            rows * n * outputs,
            best(1),
            best(4)
        );
    }
}
