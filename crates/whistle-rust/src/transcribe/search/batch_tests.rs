// The beam search on the real model with the batched decoder step (`DecoderState::step_batch`,
// what `Scorer for DecoderState` uses) against the same search stepping every hypothesis alone
// (`Alone`, the trait's default): on every speech clip at 3 and 5 beams, the finished
// hypotheses (tokens, scores to the bit, ended or cut) must be identical, so batching changes
// only the speed. Needs the private search and the model's private fields; uses
// models/whistle.cact and test-data/clips/ from the workspace root.

use std::path::Path;
use std::sync::OnceLock;

use super::{beam_search, Finished, Ranking, Scorer, Search};
use crate::audio::{features, read_wav, SAMPLE_RATE};
use crate::error::Error;
use crate::model::{DecoderState, Encoder};
use crate::transcribe::Whistle;

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

/// A decoder state that steps each hypothesis on its own (the trait's default `step_batch`).
#[derive(Clone)]
struct Alone<'w>(DecoderState<'w>);

impl Scorer for Alone<'_> {
    fn step(&mut self, token: u32) -> Result<Vec<f32>, Error> {
        self.0.step(token)
    }
}

fn no_check(_: usize) -> Result<(), Error> {
    Ok(())
}

/// The finished hypotheses as comparable values: tokens, the score's bits, ended.
fn key(finished: &[Finished]) -> Vec<(Vec<u32>, u64, bool)> {
    finished
        .iter()
        .map(|f| (f.tokens.clone(), f.score.to_bits(), f.ended))
        .collect()
}

#[test]
fn batched_beam_search_is_identical_to_stepping_each_hypothesis_alone() {
    let wh = whistle();
    let w = &wh.weights;
    let max = w.geometry.max_seq_len as usize;
    let clips = [
        "de_anna",
        "en_daniel",
        "en_samantha",
        "es_monica",
        "fr_thomas",
        "it_alice",
        "nl_xander",
        "pl_zosia",
    ];
    for clip in clips {
        let wav = std::fs::read(root().join(format!("test-data/clips/{clip}.wav"))).expect("clip");
        let audio = read_wav(&wav).expect("clip decodes");
        assert_eq!(audio.sample_rate, SAMPLE_RATE, "{clip}");
        let feats = features(&audio.samples);
        let enc = Encoder::new(w)
            .forward_with(&feats, &mut || false)
            .expect("encoder");
        let memory = w
            .decoder_memory(&enc.data, enc.frames, enc.valid_frames)
            .expect("memory");
        let mut state = DecoderState::new(w, &memory, max).expect("state");
        let scores = state.step(wh.tokenizer.bos_id()).expect("BOS");
        let tag = wh.tags[wh.detect_language(&scores)];
        for beams in [3, 5] {
            let search = Search {
                beams,
                eos: wh.tokenizer.eos_id(),
                text_ids: wh.first_tag,
                max_len: max,
                trie: None,
                ranking: Ranking::Mean,
            };
            let batched = beam_search(&search, beams, state.clone(), tag, &mut no_check)
                .expect("batched search");
            let alone = beam_search(&search, beams, Alone(state.clone()), tag, &mut no_check)
                .expect("search alone");
            assert!(!batched.is_empty(), "{clip} at {beams} beams");
            assert_eq!(key(&batched), key(&alone), "{clip} at {beams} beams");
        }
    }
}
