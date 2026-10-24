// Shared data for the decoding tests: the model bound once, the eight contract speech clips
// (WAV, the text they were synthesized from in oracle/clips.json, the engine's transcript in
// test-data/engine/), the extra keyword clips (test-data/clips-extra/, their keywords and the
// engine's transcripts with and without them in test-data/engine-extra/), and a re-scorer that
// averages a transcript's log-probabilities. The word error rate is in `wer.rs`.

use std::sync::OnceLock;

use serde_json::Value;
use whistle::audio::{features, read_wav};
use whistle::model::{DecoderState, Encoder};
use whistle::{Options, Transcript, VadMode, Whistle};

use crate::support::archive::model_bytes;
use crate::support::{read_json, repo_root};

pub fn whistle() -> &'static Whistle {
    static WHISTLE: OnceLock<Whistle> = OnceLock::new();
    WHISTLE.get_or_init(|| Whistle::from_cact(&model_bytes()).expect("whistle.cact loads"))
}

pub struct Clip {
    pub name: String,
    pub wav: Vec<u8>,
    /// The text `say` spoke.
    pub truth: String,
    /// The engine's transcript (for extra clips, without keywords).
    pub engine: String,
    /// The extra clips' keywords and the engine's transcript with them.
    pub keywords: Vec<String>,
    pub engine_keywords: String,
}

fn read(path: &str) -> Vec<u8> {
    let path = repo_root().join(path);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn engine_text(path: &str) -> String {
    read_json(&repo_root().join(path))["text"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

/// The eight contract speech clips.
pub fn contract_clips() -> Vec<Clip> {
    let list = read_json(&repo_root().join("oracle/clips.json"));
    list.as_array()
        .expect("clips.json is a list")
        .iter()
        .map(|c| {
            let name = c["name"].as_str().unwrap().to_string();
            Clip {
                wav: read(&format!("test-data/clips/{name}.wav")),
                truth: c["text"].as_str().unwrap().to_string(),
                engine: engine_text(&format!("test-data/engine/{name}.json")),
                keywords: Vec::new(),
                engine_keywords: String::new(),
                name,
            }
        })
        .collect()
}

/// The extra clips with names and jargon.
pub fn extra_clips() -> Vec<Clip> {
    let list = read_json(&repo_root().join("test-data/clips-extra/clips.json"));
    list.as_array()
        .expect("clips-extra/clips.json is a list")
        .iter()
        .map(|c: &Value| {
            let name = c["name"].as_str().unwrap().to_string();
            Clip {
                wav: read(&format!("test-data/clips-extra/{name}.wav")),
                truth: c["text"].as_str().unwrap().to_string(),
                engine: engine_text(&format!("test-data/engine-extra/{name}.json")),
                keywords: c["keywords"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|k| k.as_str().unwrap().to_string())
                    .collect(),
                engine_keywords: engine_text(&format!(
                    "test-data/engine-extra/{name}.keywords.json"
                )),
                name,
            }
        })
        .collect()
}

pub fn transcribe(clip: &Clip, options: &Options) -> Transcript {
    whistle().transcribe_wav(&clip.wav, options).unwrap()
}

pub fn beams(n: usize) -> Options {
    Options {
        beams: n,
        ..Options::default()
    }
}

/// The mean log-probability per token (log-softmax over the text ids) of `tokens` then EOS,
/// after BOS and `language`'s tag, on the whole clip: what the beam search ranks by.
pub fn log_probability(clip: &Clip, language: &str, tokens: &[u32]) -> f64 {
    let w = whistle().weights();
    let t = whistle().tokenizer();
    let audio = read_wav(&clip.wav).unwrap();
    let enc = Encoder::new(w).forward(&features(&audio.samples)).unwrap();
    let mem = w
        .decoder_memory(&enc.data, enc.frames, enc.valid_frames)
        .unwrap();
    let mut state = DecoderState::new(w, &mem, w.geometry.max_seq_len as usize).unwrap();
    state.step(t.bos_id()).unwrap();
    let mut feed = t.language_tag(language).unwrap();
    // Text ids are those below the first language tag.
    let first_tag = t.language_tags().iter().map(|&(_, id)| id).min().unwrap() as usize;
    let mut total = 0.0;
    for &next in tokens.iter().chain(std::iter::once(&t.eos_id())) {
        let scores = state.step(feed).unwrap();
        let text = &scores[..first_tag];
        let max = text.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = max
            + text
                .iter()
                .map(|&s| (s as f64 - max).exp())
                .sum::<f64>()
                .ln();
        total += text[next as usize] as f64 - lse;
        feed = next;
    }
    total / (tokens.len() + 1) as f64
}

/// Options for scoring the whole clip as `log_probability` does.
pub fn whole_clip(beams: usize) -> Options {
    Options {
        beams,
        vad: VadMode::Off,
        ..Options::default()
    }
}
