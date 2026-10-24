// The silence gate: with the default options (`VadMode::GateAndTrim`) and with `VadMode::Gate`,
// the silent clip transcribes to nothing, with no language, because the detector finds no
// speech and the decoder never runs (left alone, it says "Thank you."). With `VadMode::Off`
// the decoder runs and may say anything, so that is not pinned. Empty audio is an empty
// transcript in every mode, not an error.

use whistle::{Options, VadMode};

use crate::shared::{wav, whistle};

#[test]
fn silence_gives_an_empty_transcript_with_the_gate_on() {
    for vad in [VadMode::GateAndTrim, VadMode::Gate] {
        let options = Options {
            vad,
            beams: 1,
            ..Options::default()
        };
        let got = whistle().transcribe_wav(&wav("silence"), &options).unwrap();
        assert_eq!(got.text, "", "{vad:?}");
        assert_eq!(got.language, None, "{vad:?}");
        assert!(got.tokens.is_empty(), "{vad:?}");
    }
}

#[test]
fn empty_audio_gives_an_empty_transcript() {
    for vad in [VadMode::Off, VadMode::Gate, VadMode::GateAndTrim] {
        let options = Options {
            vad,
            beams: 1,
            ..Options::default()
        };
        let got = whistle().transcribe(&[], &options).unwrap();
        assert_eq!((got.text.as_str(), got.language), ("", None), "{vad:?}");
    }
}
