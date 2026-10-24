// The contract clips streamed in 1 s chunks against the offline decode with the same options,
// the same clips in 0.25 s and 3 s chunks against 1 s, and the words agreement commits before
// the stream ends, with their commit latency.

use whistle::{Options, VadMode};

use crate::run::{char_distance, feed, greedy, joined, latencies, median, CLIPS};
use crate::shared::{clip, whistle, RATE};
use crate::wer::wer;

#[test]
fn streamed_clips_match_the_offline_transcripts() {
    let options = greedy();
    let mut worst = 0.0f64;
    for name in CLIPS {
        let samples = clip(name);
        let offline = whistle().transcribe(&samples, &options).expect("decodes");
        let fed = feed(&samples, 1.0, &options);
        let text = joined(&fed);
        let rate = wer(&offline.text, &text);
        println!(
            "{name:12} wer {rate:.3}\n  offline  {}\n  streamed {text}",
            offline.text
        );
        worst = worst.max(rate);
        // Nothing lost or repeated, to the character.
        assert_eq!(char_distance(&offline.text, &text), 0, "{name}");
    }
    assert!(worst <= 0.0, "worst word error rate {worst:.3}");
}

#[test]
fn chunk_size_does_not_change_the_transcript() {
    let options = greedy();
    for name in ["en_daniel", "fr_thomas"] {
        let samples = clip(name);
        let one = joined(&feed(&samples, 1.0, &options));
        for chunk in [0.25, 3.0] {
            assert_eq!(
                joined(&feed(&samples, chunk, &options)),
                one,
                "{name} at {chunk} s"
            );
        }
    }
}

#[test]
fn a_stream_with_the_detector_off_still_commits_the_clip() {
    let options = Options {
        vad: VadMode::Off,
        ..greedy()
    };
    let samples = clip("en_samantha");
    let offline = whistle().transcribe(&samples, &options).expect("decodes");
    let text = joined(&feed(&samples, 1.0, &options));
    let rate = wer(&offline.text, &text);
    println!("offline  {}\nstreamed {text}", offline.text);
    assert!(rate <= 0.0, "word error rate {rate:.3}");
}

#[test]
fn agreement_commits_words_before_the_stream_ends() {
    // Every clip of 2.5 s or more has a word committed by agreement, before `finish`.
    // Measured exceptions at landing: es_monica (2.63 s) commits nothing before the end, as the
    // engine's own stream (one update, at the end); pl_zosia (2.51 s) has its first word read
    // differently by the 1 s and 2 s passes, so agreement waits for the end.
    const LATE: [&str; 2] = ["es_monica", "pl_zosia"];
    let mut all = Vec::new();
    let mut clips_early = 0;
    for name in CLIPS {
        let samples = clip(name);
        let fed = feed(&samples, 1.0, &greedy());
        let early: usize = fed.updates[..fed.updates.len() - 1]
            .iter()
            .map(|u| u.committed.len())
            .sum();
        let seconds = samples.len() as f32 / RATE as f32;
        let lat = latencies(&fed);
        println!(
            "{name:12} {seconds:.2} s, {early} early words, median latency {:.2} s",
            median(lat.clone())
        );
        if seconds >= 2.5 && !LATE.contains(&name) {
            assert!(early > 0, "{name}: nothing committed before finish");
        }
        clips_early += usize::from(early > 0);
        all.extend(lat);
    }
    // Measured: 5 of the 8 clips (the 2.27 s it_alice is too short as well).
    assert!(clips_early >= 5, "{clips_early} clips committed early");
    let latency = median(all);
    println!("median commit latency {latency:.2} s");
    // Measured: 1.26 s (1.31 s before the deterministic math; the engine's stream: 1.15 s).
    // The bound is the first measurement plus 0.15 s, one rounding of the 1 s step's agreement.
    assert!(
        latency <= 1.31 + 0.15,
        "median commit latency {latency:.2} s"
    );
}
