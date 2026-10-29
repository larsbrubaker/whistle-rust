// The per-update compute time of a stream in 1 s steps (ignored: a measurement, run in release
// with `--ignored --nocapture`): a 3 s clip and a 45 s stream, median and max, at 1 and 3 beams.
// The wasm column comes from `examples/stream.rs` built for wasm32-wasip1 and run under
// `scripts/wasm-bench.mjs`.

use std::time::Duration;

use whistle::Options;

use crate::run::feed;
use crate::shared::{clip, long_english};

fn report(label: &str, times: &[Duration]) {
    let mut ms: Vec<f64> = times.iter().map(|t| t.as_secs_f64() * 1000.0).collect();
    ms.sort_by(f64::total_cmp);
    let median = ms[ms.len() / 2];
    let max = ms.last().copied().unwrap_or(0.0);
    println!(
        "{label:28} updates {:3}  median {median:7.1} ms  max {max:7.1} ms",
        ms.len()
    );
}

/// Measured on an M5, one thread, release, with another build running:
///
/// | | native, 1 beam | native, 3 beams | wasm (Node, simd128), 1 beam | wasm, 3 beams |
/// |---|---|---|---|---|
/// | 3 s clip, median / max | 107 / 158 ms | 204 / 265 ms | 172 / 289 ms | 404 / 605 ms |
/// | 45 s stream, median / max | 225 / 446 ms | 413 / 965 ms | 361 / 954 ms | 556 / 1456 ms |
///
/// The max is a pass over a long utterance near the 30 s cap. At 3 beams a one-thread wasm
/// update can exceed the 1 s step and fall behind; at 1 beam it stays under it, which is why
/// `Options::streaming()` uses one beam.
#[test]
#[ignore = "measurement; run in release with --nocapture"]
fn measure_update_cost() {
    let short = clip("en_daniel");
    let (long, _) = long_english(12);
    for beams in [1, 3] {
        let options = Options {
            beams,
            ..Options::default()
        };
        report(
            &format!("3 s clip, {beams} beam(s)"),
            &feed(&short, 1.0, &options).times,
        );
        report(
            &format!("45 s stream, {beams} beam(s)"),
            &feed(&long, 1.0, &options).times,
        );
    }
}
