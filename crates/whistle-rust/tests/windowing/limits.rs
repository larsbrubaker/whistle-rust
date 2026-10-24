// The boundaries: up to 30 s is still one pass (the window is the single pass's span, so every
// clip decodes exactly as before, which `transcribe/contract.rs` pins against the fixtures);
// 30 s of audio is one window and one sample more is two; over 10 minutes is `Error::Input` in
// every mode; and the gate on a long input of silence gives an empty transcript.

use whistle::{plan_windows, Error, Options, VadMode, MAX_INPUT_SAMPLES, MAX_SAMPLES};

use crate::shared::{clip, greedy, whistle};
use crate::support::clip_names;

#[test]
fn every_contract_clip_is_one_window_over_the_single_pass_span() {
    for name in clip_names() {
        let samples = clip(&name);
        let off = plan_windows(&samples, VadMode::Off);
        assert_eq!(off.len(), 1, "{name}");
        assert_eq!(off[0].decode, 0..samples.len(), "{name}");
        let segments = whistle::vad::speech_segments(&samples);
        let trim = plan_windows(&samples, VadMode::GateAndTrim);
        match (segments.first(), segments.last()) {
            (Some(first), Some(last)) => {
                assert_eq!(trim.len(), 1, "{name}");
                assert_eq!(trim[0].decode, first.start..last.end, "{name}");
            }
            _ => assert!(trim.is_empty(), "{name}"),
        }
    }
}

#[test]
fn thirty_seconds_is_one_window_and_one_sample_more_is_two() {
    let speech = clip("en_samantha");
    let samples: Vec<f32> = speech
        .iter()
        .copied()
        .cycle()
        .take(MAX_SAMPLES + 1)
        .collect();
    assert_eq!(plan_windows(&samples[..MAX_SAMPLES], VadMode::Off).len(), 1);
    assert_eq!(plan_windows(&samples, VadMode::Off).len(), 2);
}

#[test]
fn over_ten_minutes_is_an_input_error_in_every_mode() {
    let long = vec![0.0f32; MAX_INPUT_SAMPLES + 1];
    for vad in [VadMode::Off, VadMode::Gate, VadMode::GateAndTrim] {
        let options = Options {
            vad,
            ..Options::default()
        };
        let err = whistle().transcribe(&long, &options).unwrap_err();
        assert!(
            matches!(&err, Error::Input(m) if m.contains("600 s")),
            "{vad:?} {err}"
        );
    }
}

#[test]
fn a_long_silence_with_the_gate_on_is_an_empty_transcript() {
    let silence = vec![0.0f32; 45 * 16_000];
    assert!(plan_windows(&silence, VadMode::Gate).is_empty());
    let got = whistle()
        .transcribe(&silence, &greedy(VadMode::Gate, true))
        .unwrap();
    assert!(got.text.is_empty() && got.tokens.is_empty() && got.words.is_empty());
    assert_eq!(got.language, None);
}
