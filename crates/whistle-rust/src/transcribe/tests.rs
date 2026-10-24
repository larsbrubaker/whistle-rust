// Unit tests for the transcription call's cancellation checks and early limits, which need the
// private per-stage hook (`Whistle::run`, `Stage`): a `Cancel` fired at any stage (after
// resampling, the detector, the features, before an encoder layer, before a decoder step, from
// another thread) stops the call at that check with `Error::Cancelled` and runs nothing after
// it; and 10 minutes of 44.1 kHz audio is rejected before it is resampled. Uses
// models/whistle.cact and test-data/clips/ from the workspace root.

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

fn whistle() -> &'static Whistle {
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
        ..Options::default()
    };
    let mut seen = Vec::new();
    let result = whistle().run(Cow::Owned(samples), rate, &options, &mut |stage| {
        if stage == at && !seen.contains(&at) {
            let handle = cancel.clone();
            std::thread::spawn(move || handle.cancel())
                .join()
                .expect("the cancelling thread runs");
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
fn ten_minutes_at_44_1_khz_with_vad_off_is_rejected_before_resampling() {
    let long = vec![0.0f32; 600 * 44_100];
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
        matches!(&result, Err(Error::Input(m)) if m.contains("30 s")),
        "{result:?}"
    );
    assert!(seen.is_empty(), "{seen:?}");
    println!("rejected in {:.1} ms", elapsed.as_secs_f64() * 1e3);
    assert!(elapsed.as_millis() < 100, "{elapsed:?}");
}
