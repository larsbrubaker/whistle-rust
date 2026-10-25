// Where the stream's 1 s passes fall relative to the speech: the contract clips streamed after
// leading silence of varying length, so each pass boundary lands at another point of the words,
// as a live recording's does (it starts whenever the user presses record), and with a faint
// hiss, as a microphone adds. `committed_words_re_read_differently_are_not_committed_again`
// pins the offsets where a pass re-read a committed word with other characters and the stream
// committed it twice; `sweep_offsets` (ignored, a measurement) prints, per clip, the runs whose
// committed text differs from the offline decode, those with an extra word, and the word error
// rate against the engine's stream. The figures in docs/design/streaming.md come from
//   SWEEP_CLIP=en_samantha SWEEP_STEP=50 SWEEP_NOISE=0.01 (or 0.02) cargo test -p whistle-rust \
//     --test streaming sweep_offsets -- --ignored --nocapture
// (320 offsets); the all-clip comparison from the defaults (SWEEP_STEP=500, every clip) at
// SWEEP_NOISE 0, 0.01 and 0.02.

use serde_json::Value;

use crate::run::{char_distance, doubled, feed, greedy, joined, CLIPS};
use crate::shared::{clip, whistle};
use crate::support::{read_json, repo_root};
use crate::wer::{normalize, wer};

/// `samples` after `pad` samples of digital silence, with a deterministic hiss of peak-to-peak
/// `noise` over all of it (none at 0).
pub fn padded(samples: &[f32], pad: usize, noise: f32) -> Vec<f32> {
    let mut out = vec![0.0; pad];
    out.extend_from_slice(samples);
    if noise > 0.0 {
        // xorshift32, seeded by the pad, so every run hears the same hiss.
        let mut state = 0x2545_f491_u32 ^ pad as u32;
        for x in &mut out {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *x += noise * (state as f32 / u32::MAX as f32 - 0.5);
        }
    }
    out
}

fn engine_text(name: &str) -> String {
    let v: Value = read_json(&repo_root().join(format!("test-data/engine-stream/{name}.json")));
    let parts: Vec<&str> = v["updates"]
        .as_array()
        .expect("updates")
        .iter()
        .filter_map(|u| u["text"].as_str())
        .filter(|t| !t.is_empty())
        .collect();
    parts.join(" ")
}

#[test]
fn committed_words_re_read_differently_are_not_committed_again() {
    // Measured before the fix (en_samantha, 0.25 s pushes): an early pass read "cube 20" as
    // "Q20" and agreement committed it; the final pass read "queue 20 mm" (or "cute 20 mm"),
    // and the alignment, finding "q20" cheaper to delete than to stretch over "queue20",
    // committed "queue 20" again: "Make the Q20 queue 20 mm wide…". 30 of 320 offsets with a
    // 0.01 hiss, 75 of 320 with 0.02; these are four of them.
    let samples = clip("en_samantha");
    for (pad, noise) in [(2800, 0.01), (3850, 0.01), (5000, 0.01), (1800, 0.02)] {
        let text = joined(&feed(&padded(&samples, pad, noise), 0.25, &greedy()));
        let words = normalize(&text);
        let twenties = words.iter().filter(|w| w.contains("20")).count();
        assert_eq!(twenties, 1, "pad {pad}, hiss {noise}: {text}");
        assert!(doubled(&text).is_empty(), "pad {pad}, hiss {noise}: {text}");
        assert!(
            words.ends_with(&["wide", "and", "round", "the", "top", "edges"].map(String::from)),
            "pad {pad}, hiss {noise}: {text}"
        );
    }
}

#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn sweep_offsets() {
    // SWEEP_STEP (samples between pads, default 500), SWEEP_CLIP (one clip), SWEEP_NOISE (the
    // hiss, default none).
    let env = |name: &str| std::env::var(name).ok();
    let step: usize = env("SWEEP_STEP")
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let noise: f32 = env("SWEEP_NOISE")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let only = env("SWEEP_CLIP");
    let options = greedy();
    let (mut runs, mut differ, mut extra, mut total_wer) = (0, 0, 0, 0.0);
    for name in CLIPS {
        if only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        let samples = clip(name);
        let engine = engine_text(name);
        let (mut clip_differ, mut clip_extra, mut clip_wer, mut count) = (0, 0, 0.0, 0);
        for pad in (0..16_000).step_by(step) {
            let audio = padded(&samples, pad, noise);
            let offline = whistle().transcribe(&audio, &options).expect("decodes");
            let text = joined(&feed(&audio, 0.25, &options));
            let distance = char_distance(&offline.text, &text);
            let rate = wer(&engine, &text);
            // A word committed twice shows as an extra word (or a doubled pair).
            let extra = normalize(&text).len() > normalize(&offline.text).len()
                || !doubled(&text).is_empty();
            clip_extra += usize::from(extra);
            if distance > 0 {
                clip_differ += 1;
                println!(
                    "{name} pad {pad}: distance {distance} wer {rate:.3}\n  offline  {}\n  streamed {text}",
                    offline.text
                );
            }
            clip_wer += rate;
            count += 1;
        }
        println!(
            "{name:12} {clip_differ} of {count} differ from offline, {clip_extra} with an extra word, mean wer vs engine {:.3}",
            clip_wer / count as f64
        );
        runs += count;
        differ += clip_differ;
        extra += clip_extra;
        total_wer += clip_wer;
    }
    println!(
        "all: {differ} of {runs} differ, {extra} with an extra word, mean wer vs engine {:.4}",
        total_wer / runs as f64
    );
}
