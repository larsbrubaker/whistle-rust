// The errors a caller can see from `Whistle::transcribe`: an unsupported language code (the
// message lists the supported ones), audio over the 10-minute limit (the message says it), and
// cancellation fired before the call. Cancellation part way through (between stages, encoder
// layers and decoder steps) is pinned deterministically by the unit tests in
// `src/transcribe/tests.rs`; the inputs rejected before decoding are in `inputs`.

use whistle::{Cancel, Error, Options, VadMode};

use crate::shared::{wav, whistle};

#[test]
fn an_unknown_language_is_an_input_error_listing_the_supported_codes() {
    let options = Options {
        language: Some("xx".into()),
        ..Options::default()
    };
    let err = whistle().transcribe(&[0.0; 16000], &options).unwrap_err();
    let Error::Input(message) = &err else {
        panic!("{err}");
    };
    for code in ["en", "de", "fr", "es", "it", "nl", "pl"] {
        assert!(message.contains(code), "{message}");
    }
}

#[test]
fn audio_over_ten_minutes_is_an_input_error_naming_the_limit() {
    let samples = vec![0.0f32; whistle::MAX_INPUT_SAMPLES + 1];
    let options = Options {
        vad: VadMode::Off,
        ..Options::default()
    };
    let err = whistle().transcribe(&samples, &options).unwrap_err();
    assert!(
        matches!(&err, Error::Input(m) if m.contains("600 s")),
        "{err}"
    );
}

#[test]
fn a_cancel_fired_before_the_call_gives_cancelled() {
    let cancel = Cancel::new();
    let handle = cancel.clone();
    handle.cancel();
    assert!(cancel.is_cancelled());
    let options = Options {
        cancel: Some(cancel),
        ..Options::default()
    };
    let err = whistle()
        .transcribe_wav(&wav("en_samantha"), &options)
        .unwrap_err();
    assert_eq!(err, Error::Cancelled);
}
