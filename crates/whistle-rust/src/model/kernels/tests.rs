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
