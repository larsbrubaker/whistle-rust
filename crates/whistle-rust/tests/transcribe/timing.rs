// The cost of one transcription: the encoder, the cached decoder per token and the whole call
// on en_samantha, and the stem and each encoder layer on 30 s, each on one worker thread and on
// the default count (`whistle::model::with_threads`, per calling thread), printed for
// `docs/design/performance.md`. Run in release for the real numbers:
// `cargo test --release --test transcribe timing -- --include-ignored --nocapture --test-threads=1`.
// It asserts nothing about speed; it only checks the clip still transcribes. Measured
// 2026-10-24 on en_samantha (3.09 s), release, Apple Silicon, default threads: the encoder
// 31 ms, the decoder setup (cross-attention memory for 8 layers) 3.4 ms, then about 2.7 ms per
// decoder token (one thread whatever the setting: the few-row kernel path), the whole one-beam
// call (trimmed to the speech span) 81–87 ms.

use std::time::Instant;

use whistle::audio::{features, read_wav};
use whistle::model::{DecoderState, Encoder};
use whistle::Options;

use crate::shared::{wav, weights, whistle};
use crate::support::Fixture;

#[test]
fn timing_of_one_clip() {
    for threads in [1, 0] {
        whistle::model::with_threads(threads, one_clip);
    }
}

fn one_clip() {
    let w = weights();
    let audio = read_wav(&wav("en_samantha")).unwrap();
    let start = Instant::now();
    let feats = features(&audio.samples);
    let enc = Encoder::new(w).forward(&feats).unwrap();
    let encoder_ms = start.elapsed().as_secs_f64() * 1e3;

    let tokens = Fixture::load("en_samantha").tokens();
    let start = Instant::now();
    let mem = w
        .decoder_memory(&enc.data, enc.frames, enc.valid_frames)
        .unwrap();
    let mut state = DecoderState::new(w, &mem, w.geometry.max_seq_len as usize).unwrap();
    let setup_ms = start.elapsed().as_secs_f64() * 1e3;
    let start = Instant::now();
    for &t in &tokens {
        state.step(t).unwrap();
    }
    let per_token_ms = start.elapsed().as_secs_f64() * 1e3 / tokens.len() as f64;

    let start = Instant::now();
    let got = whistle()
        .transcribe_wav(
            &wav("en_samantha"),
            &Options {
                beams: 1,
                ..Options::default()
            },
        )
        .unwrap();
    let total_ms = start.elapsed().as_secs_f64() * 1e3;
    println!(
        "{} thread(s), en_samantha ({:.2} s): encoder {encoder_ms:.1} ms, decoder setup {setup_ms:.1} ms, \
         decoder {per_token_ms:.2} ms per token ({} tokens), whole call {total_ms:.1} ms",
        whistle::threads(),
        audio.samples.len() as f64 / audio.sample_rate as f64,
        tokens.len()
    );
    assert!(!got.text.is_empty());
}

/// The cancellation latency on 30 s of audio (en_samantha repeated), which is the longest gap
/// between two checks: the front end, the stem before the first encoder layer, and each encoder
/// layer. Run in release: `cargo test --release --test transcribe -- --ignored --nocapture`.
#[test]
#[ignore = "release measurement of the cancellation latency"]
fn the_longest_gap_between_cancellation_checks_on_thirty_seconds() {
    for threads in [1, 0] {
        whistle::model::with_threads(threads, thirty_seconds);
    }
}

fn thirty_seconds() {
    let w = weights();
    let clip = read_wav(&wav("en_samantha")).unwrap().samples;
    let samples: Vec<f32> = clip.iter().copied().cycle().take(480_000).collect();
    let start = Instant::now();
    let feats = features(&samples);
    let features_ms = start.elapsed().as_secs_f64() * 1e3;
    let start = Instant::now();
    let vad_ms = {
        whistle::vad::speech_segments(&samples);
        start.elapsed().as_secs_f64() * 1e3
    };
    let mut marks = vec![Instant::now()];
    Encoder::new(w)
        .forward_with(&feats, &mut || {
            marks.push(Instant::now());
            false
        })
        .unwrap();
    marks.push(Instant::now());
    let gaps: Vec<f64> = marks
        .windows(2)
        .map(|p| (p[1] - p[0]).as_secs_f64() * 1e3)
        .collect();
    let total_ms = features_ms + gaps.iter().sum::<f64>();
    println!(
        "{} thread(s), 30 s: features {features_ms:.0} ms, VAD {vad_ms:.0} ms; stem {:.0} ms; encoder layers \
         {:?} ms (the last includes the final norm); features to encoder output {total_ms:.0} ms",
        whistle::threads(),
        gaps[0],
        gaps[1..].iter().map(|g| g.round()).collect::<Vec<_>>()
    );
}
