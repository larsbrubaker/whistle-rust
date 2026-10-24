// The cost of a windowed transcription (ignored; run in release with `--ignored --nocapture`):
// the ~69 s English input with pauses, its window sizes, and the time per call at one beam and
// at the default three, with and without word times. Prints the numbers in
// docs/design/windowing.md.

use std::time::Instant;

use whistle::{plan_windows, Options, VadMode};

use crate::shared::{long_english, seconds, whistle};

#[test]
#[ignore = "measurement for docs/design/windowing.md"]
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
