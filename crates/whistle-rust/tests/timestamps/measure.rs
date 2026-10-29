// Ignored measurements behind word timestamps (`src/transcribe/align.rs`): the release cost of
// `word_timestamps: true` on one clip, each clip's timing errors against the engine, and our
// word probabilities beside the engine's.

use whistle::{Options, FRAME_SECONDS};

use crate::shared::{engine, speech_clips, timed, wav, whistle};
use crate::support::timed_runs_ms;

/// The extra time `word_timestamps: true` costs on en_daniel (3.4 s) at the default options.
/// Run in release: `cargo test --release -p whistle-rust --test timestamps
/// cost_of_word_timestamps -- --ignored --nocapture`. Each setting is timed over 7 calls and
/// reported as their median, so a call slowed by other work on the machine does not move the
/// figure.
///
/// Measured 2026-10-29, release, Apple Silicon, the default kernel threads (up to 4), on a
/// machine shared with other builds (load average 21-35 on 10 cores): the extra per call, in
/// six runs of this test, was +25, +37, +30, +77, +58 and -5 ms, a median of +34 ms (off
/// 141-267 ms, on 172-278 ms).
#[test]
#[ignore = "measurement: run in release with --nocapture"]
fn cost_of_word_timestamps() {
    let bytes = wav("en_daniel");
    let cost = |options: &Options| {
        whistle()
            .transcribe_wav(&bytes, options)
            .expect("transcribes");
        timed_runs_ms(7, || {
            whistle()
                .transcribe_wav(&bytes, options)
                .expect("transcribes");
        })
    };
    let ((off_runs, off), (on_runs, on)) = (cost(&Options::default()), cost(&timed()));
    println!(
        "en_daniel: off median {off:.1} ms (runs {off_runs:.0?}), on median {on:.1} ms \
         (runs {on_runs:.0?}), +{:.1} ms",
        on - off
    );
}

#[test]
#[ignore = "measurement: run with --nocapture"]
fn timing_errors_and_probabilities_per_clip() {
    let (mut all, mut probs) = (Vec::new(), Vec::new());
    for clip in speech_clips() {
        let t = whistle()
            .transcribe_wav(&wav(&clip), &timed())
            .expect("transcribes");
        let mut errs = Vec::new();
        for (w, e) in t.words.iter().zip(engine("engine", &clip).1) {
            errs.push(((w.start - e.start) / FRAME_SECONDS).round() as i32);
            errs.push(((w.end - e.end) / FRAME_SECONDS).round() as i32);
            probs.push((w.probability, e.probability));
        }
        let exact = errs.iter().filter(|&&e| e == 0).count();
        println!(
            "{clip}: {exact}/{} exact, frame errors {errs:?}",
            errs.len()
        );
        all.extend(errs);
    }
    for d in -2..=2 {
        println!(
            "error {d:+} frames: {}",
            all.iter().filter(|&&e| e == d).count()
        );
    }
    let n = probs.len() as f32;
    let mae = probs.iter().map(|(a, b)| (a - b).abs()).sum::<f32>() / n;
    let mean = |f: fn(&(f32, f32)) -> f32| probs.iter().map(f).sum::<f32>() / n;
    println!(
        "probability: mean ours {:.3}, engine {:.3}, mean |diff| {mae:.3}",
        mean(|p| p.0),
        mean(|p| p.1)
    );
    for (a, b) in &probs {
        print!("{a:.3}/{b:.3} ");
    }
    println!();
}
