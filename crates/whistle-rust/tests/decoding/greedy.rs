// One beam is the greedy decode: on every contract speech clip `beams = 1` teacher-forces to the
// manifest's tokens (the contract, `support::forced`, near-ties listed) and free-runs to the
// committed snapshot of the port's own output (test-data/snapshots/greedy.json, identical on
// every target); the default (three beams) free-runs to test-data/snapshots/default.json and
// gives the engine's transcript on every clip; and the beam and keyword options reject values a
// caller could not mean with `Error::Input`. `regenerate_snapshots` (ignored) rewrites both
// snapshots when a change deliberately moves the port's output.

use serde_json::Value;
use whistle::{Error, Options, VadMode, MAX_BEAMS};

use crate::clips::{beams, contract_clips, transcribe, whistle};
use crate::support::forced::{check_contract, free_running, snapshot};
use crate::support::repo_root;

fn clips() -> Vec<(String, Vec<u8>)> {
    contract_clips()
        .into_iter()
        .map(|c| (c.name, c.wav))
        .collect()
}

/// The pinned snapshot, entry by entry so a failure names the clip.
fn assert_snapshot(name: &str, got: &Value) {
    let want = snapshot(name);
    for (clip, entry) in want.as_object().expect("a snapshot is an object") {
        assert_eq!(&got[clip], entry, "{name}: {clip}");
    }
    assert_eq!(got, &want, "{name}: the clip list changed");
}

#[test]
fn one_beam_gives_the_greedy_contract_tokens_on_every_clip() {
    check_contract(whistle(), &clips(), VadMode::GateAndTrim);
    assert_snapshot("greedy", &free_running(whistle(), &clips(), &beams(1)));
}

#[test]
fn the_default_options_free_run_to_the_snapshot_on_every_clip() {
    assert_snapshot(
        "default",
        &free_running(whistle(), &clips(), &Options::default()),
    );
}

/// Rewrites test-data/snapshots/ from the port's current output. The snapshots record the port,
/// not the reference: run this only with a change that deliberately moves the output, and say
/// why in its commit.
#[test]
#[ignore]
fn regenerate_snapshots() {
    let dir = repo_root().join("test-data/snapshots");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, options) in [("greedy", beams(1)), ("default", Options::default())] {
        let v = free_running(whistle(), &clips(), &options);
        let text = serde_json::to_string_pretty(&v).unwrap() + "\n";
        std::fs::write(dir.join(format!("{name}.json")), text).unwrap();
    }
}

#[test]
fn zero_beams_too_many_beams_and_a_bad_bonus_are_input_errors() {
    let clip = &contract_clips()[0];
    for options in [
        beams(0),
        beams(MAX_BEAMS + 1),
        Options {
            keyword_bonus: -1.0,
            ..Options::default()
        },
        Options {
            keyword_bonus: f32::NAN,
            ..Options::default()
        },
    ] {
        let got = whistle().transcribe_wav(&clip.wav, &options);
        assert!(matches!(got, Err(Error::Input(_))), "{options:?}: {got:?}");
    }
}

#[test]
fn the_default_is_three_beams_without_keywords() {
    let o = Options::default();
    assert_eq!(o.beams, 3);
    assert_eq!(o.beams, whistle::DEFAULT_BEAMS);
    assert!(o.keywords.is_empty());
    assert_eq!(o.keyword_bonus, whistle::DEFAULT_KEYWORD_BONUS);
}

/// The default (three beams, measured 2026-10-24) reproduces the engine's transcript on every
/// contract clip, the measurement that chose it (`whistle::DEFAULT_BEAMS`).
#[test]
fn the_default_options_give_the_engine_transcript_on_every_clip() {
    for clip in contract_clips() {
        let got = transcribe(&clip, &Options::default());
        assert_eq!(got.text, clip.engine, "{}", clip.name);
    }
}
