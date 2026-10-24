// The end-to-end contract on every speech clip (CLAUDE.md "Testing framework", net 2): from the
// WAV file through our features, encoder and cached greedy decode, the text tokens equal the
// manifest's `tokens[2..len-1]`, and the language and text equal the manifest's, with one beam
// (`beams: 1`, the greedy decode the fixtures pin; the default is a 3-beam search), the VAD
// off (`VadMode::Off`, the whole clip as the reference decodes it) and with the default
// options (`VadMode::GateAndTrim`, only the speech span). A mismatch reports the first differing
// position and the top-2 score margin there, from the full decoder on our own encoder output,
// so a near-tie can be told from a bug.

use whistle::audio::{features, read_wav};
use whistle::model::{Decoder, Encoder};
use whistle::{Options, VadMode};

use crate::shared::{speech_clips, text_tokens, wav, weights, whistle};
use crate::support::Fixture;

/// The first position where `got` leaves the manifest's stream, and the margin between the
/// best two allowed scores there in our full decode of the manifest prefix.
fn divergence(clip: &str, got: &[u32]) -> String {
    let f = Fixture::load(clip);
    let want = f.tokens();
    let ids: Vec<u32> = want[..2]
        .iter()
        .copied()
        .chain(got.iter().copied())
        .collect();
    let pos = (0..want.len().max(ids.len()))
        .find(|&i| want.get(i) != ids.get(i))
        .unwrap_or(0);
    let w = weights();
    let audio = read_wav(&wav(clip)).unwrap();
    let enc = Encoder::new(w).forward(&features(&audio.samples)).unwrap();
    let mem = w
        .decoder_memory(&enc.data, enc.frames, enc.valid_frames)
        .unwrap();
    let prefix = &want[..pos.min(want.len())];
    let logits = Decoder::new(w).forward(prefix, &mem).unwrap();
    let row = logits.row(pos - 1).unwrap();
    let mut scores: Vec<(f32, usize)> = row[..8192]
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();
    scores.sort_by(|a, b| b.0.total_cmp(&a.0));
    format!(
        "{clip}: first difference at position {pos} (manifest {:?}, ours {:?}); top two {:?} \
         and {:?}, margin {:.4}",
        want.get(pos),
        ids.get(pos),
        scores[0],
        scores[1],
        scores[0].0 - scores[1].0
    )
}

fn check_every_clip(options: &Options) {
    let mut failures = Vec::new();
    for clip in speech_clips() {
        let f = Fixture::load(&clip);
        let got = whistle().transcribe_wav(&wav(&clip), options).unwrap();
        println!("{clip}: {:?} {:?}", got.language, got.text);
        if got.tokens != text_tokens(&f.tokens()) {
            failures.push(divergence(&clip, &got.tokens));
            continue;
        }
        assert_eq!(got.language.as_deref(), Some(f.language()), "{clip}");
        assert_eq!(got.text, f.text(), "{clip}");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn every_speech_clip_transcribes_to_the_reference_tokens_with_vad_off() {
    check_every_clip(&Options {
        vad: VadMode::Off,
        beams: 1,
        ..Options::default()
    });
}

#[test]
fn every_speech_clip_transcribes_to_the_reference_text_with_the_default_trim() {
    assert_eq!(Options::default().vad, VadMode::GateAndTrim);
    check_every_clip(&Options {
        beams: 1,
        ..Options::default()
    });
}
