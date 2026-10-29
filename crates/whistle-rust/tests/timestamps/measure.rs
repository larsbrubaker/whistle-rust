// Ignored measurements behind word timestamps (`src/transcribe/align.rs`): the release cost of
// `word_timestamps: true` on one clip, each clip's timing errors against the engine, and our
// word probabilities beside the engine's.

use std::time::Instant;

use whistle::{Options, FRAME_SECONDS};

use crate::shared::{engine, speech_clips, timed, wav, whistle};

#[test]
#[ignore = "measurement: run in release with --nocapture"]
fn cost_of_word_timestamps() {
    let bytes = wav("en_daniel");
    let best = |options: &Options| {
        (0..5)
            .map(|_| {
                let t = Instant::now();
                whistle()
                    .transcribe_wav(&bytes, options)
                    .expect("transcribes");
                t.elapsed().as_secs_f64() * 1e3
            })
            .fold(f64::INFINITY, f64::min)
    };
    let (off, on) = (best(&Options::default()), best(&timed()));
    println!(
        "en_daniel: off {off:.1} ms, on {on:.1} ms, +{:.1} ms",
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
