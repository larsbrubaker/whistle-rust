// The cost of a windowed transcription (ignored; run in release with `--ignored --nocapture`):
// the ~69 s English input with pauses, its window sizes, and the time per call at one beam and
// at the default three, with and without word times. Run with
// `cargo test --release -p whistle-rust --test windowing long_input_cost -- --ignored --nocapture`.

use std::time::Instant;

use whistle::{plan_windows, Options, VadMode};

use crate::shared::{long_english, seconds, whistle};

/// Measured 2026-10-24 on Apple Silicon, release, on a machine shared with a benchmark run (so
/// the spread between the two runs is load): the ~69 s input (the five English clips in turn,
/// 18 clips, 0.5/0.75/1 s pauses) plans to 3 windows of 22.10, 22.49 and 21.34 s, each cut in a
/// 1 s pause after the 15 s mark. Per call, default `GateAndTrim`:
///
/// | beams | word times | ms per call (two runs) |
/// | --- | --- | --- |
/// | 1 | off | 3076, 2238 |
/// | 1 | on | 3876, 3634 |
/// | 3 | off | 4379, 4529 |
/// | 3 | on | 5450, 7143 |
///
/// About 33–45 ms per second of audio at one beam, most of it the encoder on 22 s windows and
/// the ~100-token decode per window.
#[test]
#[ignore = "measurement: run in release with --ignored --nocapture"]
fn long_input_cost() {
    let (samples, _) = long_english(18);
    let windows = plan_windows(&samples, VadMode::GateAndTrim);
    let sizes: Vec<String> = windows
        .iter()
        .map(|w| format!("{:.2}", seconds(w.decode.len())))
        .collect();
    println!(
        "{:.1} s input, {} windows: [{}] s",
        seconds(samples.len()),
        windows.len(),
        sizes.join(", ")
    );
    let w = whistle();
    for (beams, word_timestamps) in [(1, false), (1, true), (3, false), (3, true)] {
        let options = Options {
            beams,
            word_timestamps,
            ..Options::default()
        };
        w.transcribe(&samples, &options).unwrap();
        let runs = 3;
        let start = Instant::now();
        for _ in 0..runs {
            w.transcribe(&samples, &options).unwrap();
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(runs);
        println!("beams {beams}, word times {word_timestamps}: {ms:.0} ms per call");
    }
}
