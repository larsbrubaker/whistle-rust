// One beam is the greedy decode: on every contract speech clip `beams = 1` gives exactly the
// manifest's token stream (the contract); the default is three beams, which gives the engine's
// transcript on every clip; and the beam and keyword options reject values a caller could not
// mean with `Error::Input`.

use whistle::{Error, Options, MAX_BEAMS};

use crate::clips::{beams, contract_clips, transcribe, whistle};
use crate::support::Fixture;

#[test]
fn one_beam_gives_the_greedy_contract_tokens_on_every_clip() {
    for clip in contract_clips() {
        let f = Fixture::load(&clip.name);
        let want = f.tokens();
        let got = transcribe(&clip, &beams(1));
        assert_eq!(
            got.tokens,
            want[2..want.len() - 1].to_vec(),
            "{}",
            clip.name
        );
        assert_eq!(got.text, f.text(), "{}", clip.name);
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
/// contract clip, the measurement that chose it (docs/design/decoding.md).
#[test]
fn the_default_options_give_the_engine_transcript_on_every_clip() {
    for clip in contract_clips() {
        let got = transcribe(&clip, &Options::default());
        assert_eq!(got.text, clip.engine, "{}", clip.name);
    }
}
