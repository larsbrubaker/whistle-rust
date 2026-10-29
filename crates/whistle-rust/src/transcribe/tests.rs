// Unit tests for the transcription call's cancellation checks and early limits, which need the
// private per-stage hook (`Whistle::run`, `Stage`): a `Cancel` fired at any stage (after
// resampling, the detector, the features, before an encoder layer, before a decoder step,
// between the windows of a long input, from another thread) stops the call at that check with
// `Error::Cancelled` and runs nothing after it; and over 10 minutes of 44.1 kHz audio is
// rejected before it is resampled. Also the ignored
// measurement of the beam search's final ranking (mean against sum, which needs the private
// `Whistle::run_ranked`). Uses models/whistle.cact and test-data/ from the workspace root.

use std::borrow::Cow;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use super::{Cancel, Options, Stage, VadMode, Whistle};
use crate::audio::read_wav;
use crate::error::Error;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/whistle-rust sits two levels below the workspace root")
}

pub(crate) fn whistle() -> &'static Whistle {
    static WHISTLE: OnceLock<Whistle> = OnceLock::new();
    WHISTLE.get_or_init(|| {
        let bytes = std::fs::read(root().join("models/whistle.cact")).expect("whistle.cact");
        Whistle::from_cact(&bytes).expect("whistle.cact loads")
    })
}

fn samples() -> Vec<f32> {
    let wav = std::fs::read(root().join("test-data/clips/en_samantha.wav")).expect("clip");
    read_wav(&wav).expect("clip decodes").samples
}

/// Runs en_samantha (at `rate`, as given) with a cancel fired from another thread when the hook
/// first sees `at`; returns the result and every stage the hook saw.
fn cancel_at(samples: Vec<f32>, rate: u32, at: Stage) -> (Result<(), Error>, Vec<Stage>) {
    let cancel = Cancel::new();
    let options = Options {
        cancel: Some(cancel.clone()),
        // One beam: one decoder step per position, as the stage counts assume.
        beams: 1,
        ..Options::default()
    };
    let mut seen = Vec::new();
    let result = whistle().run(Cow::Owned(samples), rate, &options, &mut |stage| {
        if stage == at && !seen.contains(&at) {
            let handle = cancel.clone();
            // wasm32 cannot spawn a thread, so there the cancel fires on the calling thread: the
            // call still has to notice it at its next check, which is what these tests pin.
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::spawn(move || handle.cancel())
                .join()
                .expect("the cancelling thread runs");
            #[cfg(target_arch = "wasm32")]
            handle.cancel();
        }
        seen.push(stage);
    });
    (result.map(|_| ()), seen)
}

#[test]
fn a_cancel_from_another_thread_mid_decode_stops_before_the_next_step() {
    let (result, seen) = cancel_at(samples(), 16000, Stage::DecoderStep(4));
    assert_eq!(result, Err(Error::Cancelled));
    assert_eq!(seen.last(), Some(&Stage::DecoderStep(4)));
    let steps = seen
        .iter()
        .filter(|s| matches!(s, Stage::DecoderStep(_)))
        .count();
    assert_eq!(
        steps, 5,
        "steps 0 to 3 ran, then the check before step 4: {seen:?}"
    );
}

#[test]
fn a_cancel_during_the_encoder_stops_before_the_next_layer() {
    let (result, seen) = cancel_at(samples(), 16000, Stage::EncoderLayer);
    assert_eq!(result, Err(Error::Cancelled));
    // The first layer's check saw the cancel: no layer ran and the decoder never started.
    assert_eq!(
        seen,
        vec![
            Stage::Start,
            Stage::Vad,
            Stage::Features,
            Stage::EncoderLayer
        ]
    );
}

#[test]
fn a_cancel_between_windows_stops_before_the_next_window() {
    // en_samantha with half a second of silence after it, repeated past 30 s: two windows.
    let clip = samples();
    let mut long = Vec::new();
    while long.len() <= 40 * 16000 {
        long.extend_from_slice(&clip);
        long.resize(long.len() + 8000, 0.0);
    }
    let (result, seen) = cancel_at(long, 16000, Stage::Window(1));
    assert_eq!(result, Err(Error::Cancelled));
    assert_eq!(seen.last(), Some(&Stage::Window(1)), "{seen:?}");
    let first = seen.iter().position(|s| *s == Stage::Window(0));
    let done = seen.iter().position(|s| *s == Stage::Done);
    assert!(first < done, "the first window was decoded: {seen:?}");
}

#[test]
fn a_cancel_after_each_early_stage_stops_there() {
    for at in [Stage::Start, Stage::Vad, Stage::Features] {
        let (result, seen) = cancel_at(samples(), 16000, at);
        assert_eq!(result, Err(Error::Cancelled), "{at:?}");
        assert_eq!(seen.last(), Some(&at), "{seen:?}");
    }
    // Feeding 16 kHz samples as 32 kHz makes the call resample (to half the length).
    let (result, seen) = cancel_at(samples(), 32000, Stage::Resampled);
    assert_eq!(result, Err(Error::Cancelled));
    assert_eq!(seen, vec![Stage::Start, Stage::Resampled]);
}

#[test]
fn over_ten_minutes_at_44_1_khz_with_vad_off_is_rejected_before_resampling() {
    let long = vec![0.0f32; 601 * 44_100];
    let options = Options {
        vad: VadMode::Off,
        ..Options::default()
    };
    let whistle = whistle(); // loaded before the clock starts
    let start = Instant::now();
    let mut seen = Vec::new();
    let result = whistle.run(Cow::Owned(long), 44_100, &options, &mut |s| seen.push(s));
    let elapsed = start.elapsed();
    assert!(
        matches!(&result, Err(Error::Input(m)) if m.contains("600 s")),
        "{result:?}"
    );
    assert!(seen.is_empty(), "{seen:?}");
    println!("rejected in {:.1} ms", elapsed.as_secs_f64() * 1e3);
    assert!(elapsed.as_millis() < 100, "{elapsed:?}");
}

#[path = "../../tests/decoding/wer.rs"]
mod wer;

/// The beam search's final ranking measured on the contract clips: normalized WER against the
/// true texts (oracle/clips.json) and the engine's transcripts (test-data/engine/), per beam
/// count, under the mean per token (the default) and the plain sum, over 72 true words and 73
/// engine words: `cargo test --release --lib ranking -- --ignored --nocapture`. Measured
/// 2026-10-24, normalized corpus WER:
///
/// | beams | truth (sum) | engine (sum) | truth (mean) | engine (mean) |
/// |------:|------------:|-------------:|-------------:|--------------:|
/// | 1     | 0.1944      | 0.0274       | 0.1944       | 0.0274        |
/// | 2     | 0.1667      | 0.0548       | 0.1667       | 0.0274        |
/// | 3     | 0.1667      | 0.0548       | 0.1667       | 0.0000        |
/// | 5     | 0.1667      | 0.0548       | 0.1667       | 0.0000        |
/// | 8     | 0.1667      | 0.0548       | 0.1667       | 0.0000        |
///
/// The sum turns fr_thomas's "5000 mètres" into "5 mm" and keeps pl_zosia's short "do 10 ml.";
/// the mean keeps fr_thomas as greedy has it and reads pl_zosia's "do 10 milimetrów lewo", the
/// engine's.
#[test]
#[ignore = "measurement for the beam search's final ranking"]
fn ranking_word_error_rates() {
    use super::search::Ranking;
    let read = |p: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(root().join(p)).unwrap()).unwrap()
    };
    let clips = read("oracle/clips.json");
    for ranking in [Ranking::Mean, Ranking::Sum] {
        for n in [1, 2, 3, 5, 8] {
            let (mut truth_edits, mut engine_edits, mut truth_words, mut engine_words) =
                (0, 0, 0, 0);
            for c in clips.as_array().unwrap() {
                let name = c["name"].as_str().unwrap();
                let truth = c["text"].as_str().unwrap();
                let engine = read(&format!("test-data/engine/{name}.json"))["text"]
                    .as_str()
                    .unwrap()
                    .to_string();
                let bytes =
                    std::fs::read(root().join(format!("test-data/clips/{name}.wav"))).unwrap();
                let audio = read_wav(&bytes).unwrap();
                let options = Options {
                    beams: n,
                    ..Options::default()
                };
                let text = whistle()
                    .run_ranked(
                        Cow::Owned(audio.samples),
                        audio.sample_rate,
                        &options,
                        &mut |_| {},
                        ranking,
                    )
                    .unwrap()
                    .text;
                let (t, e) = (wer::normalize(truth), wer::normalize(&engine));
                let h = wer::normalize(&text);
                let (te, ee) = (wer::edits(&t, &h), wer::edits(&e, &h));
                println!(
                    "{ranking:?} beams {n} {name:12} truth {:.3} (raw {:.3}) engine {:.3}  {text}",
                    wer::wer(truth, &text),
                    wer::raw_wer(truth, &text),
                    ee as f64 / e.len().max(1) as f64
                );
                truth_edits += te;
                engine_edits += ee;
                truth_words += t.len();
                engine_words += e.len();
            }
            println!(
                "{ranking:?} beams {n}: corpus truth {truth_edits}/{truth_words} = {:.4}, engine \
                 {engine_edits}/{engine_words} = {:.4}",
                truth_edits as f64 / truth_words as f64,
                engine_edits as f64 / engine_words as f64
            );
        }
    }
}
