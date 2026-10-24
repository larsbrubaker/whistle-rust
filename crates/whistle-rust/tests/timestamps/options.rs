// What the option changes and what it does not: with timings off `words` is empty and `text`,
// `language` and `tokens` are what they are with timings on; silence has no words; and with the
// default trim the times are exactly those of decoding the trimmed samples alone, shifted by the
// samples trimmed off (one beam and three clips, to keep the gate cheap).

use whistle::vad::speech_segments;
use whistle::{Options, VadMode};

use crate::shared::{timed, wav, whistle};

#[test]
fn timings_off_leave_words_empty_and_change_nothing_else() {
    let bytes = wav("nl_xander");
    let greedy = Options {
        beams: 1,
        ..Options::default()
    };
    let off = whistle()
        .transcribe_wav(&bytes, &greedy)
        .expect("transcribes");
    let on = whistle()
        .transcribe_wav(
            &bytes,
            &Options {
                word_timestamps: true,
                ..greedy
            },
        )
        .expect("transcribes");
    assert!(off.words.is_empty(), "{off:?}");
    assert!(!on.words.is_empty(), "{on:?}");
    assert_eq!(off.text, on.text);
    assert_eq!(off.language, on.language);
    assert_eq!(off.tokens, on.tokens);
}

#[test]
fn silence_has_no_words() {
    let t = whistle()
        .transcribe_wav(&wav("silence"), &timed())
        .expect("transcribes");
    assert!(t.text.is_empty() && t.words.is_empty(), "{t:?}");
}

/// 1 s of zeros before a clip: the detector keeps `b0..b1` (its segment start pads into the
/// zeros, so the decode sees a lead-in the clean clip lacks and may decode differently,
/// `docs/design/timestamps.md`). Whatever it decodes, the times must be those of decoding
/// `b0..b1` with the detector off, plus `b0` samples, exactly.
#[test]
fn the_trim_is_added_back_to_every_time_exactly() {
    for clip in ["en_daniel", "it_alice", "pl_zosia"] {
        let audio = whistle::audio::read_wav(&wav(clip)).expect("reads");
        assert_eq!(audio.sample_rate, 16_000);
        let mut padded = vec![0.0f32; 16_000];
        padded.extend_from_slice(&audio.samples);
        let segments = speech_segments(&padded);
        let (b0, b1) = (segments[0].start, segments[segments.len() - 1].end);
        let options = |vad| Options {
            vad,
            beams: 1,
            ..timed()
        };
        let trimmed = whistle()
            .transcribe(&padded[b0..b1], &options(VadMode::Off))
            .expect("transcribes");
        let gated = whistle()
            .transcribe(&padded, &options(VadMode::GateAndTrim))
            .expect("transcribes");
        assert_eq!(gated.tokens, trimmed.tokens, "{clip}");
        assert!(!gated.words.is_empty(), "{clip}");
        let shift = b0 as f32 / 16_000.0;
        for (g, t) in gated.words.iter().zip(&trimmed.words) {
            assert_eq!(g.word, t.word, "{clip}");
            assert_eq!(
                (g.start, g.end),
                (t.start + shift, t.end + shift),
                "{clip} {g:?}"
            );
        }
    }
}
