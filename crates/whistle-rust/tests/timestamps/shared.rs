// Shared state for the timestamp test area: the model bound once per test process, the clips'
// WAV bytes (test-data/clips/ and the held-out test-data/clips-extra/), the options with timings
// on, and the engine's transcript and words for a clip.

use std::sync::OnceLock;

use whistle::{Options, Whistle};

use crate::support::archive::model_bytes;
use crate::support::{clip_names, read_json, repo_root};

pub fn whistle() -> &'static Whistle {
    static WHISTLE: OnceLock<Whistle> = OnceLock::new();
    WHISTLE.get_or_init(|| Whistle::from_cact(&model_bytes()).expect("whistle.cact loads"))
}

pub fn speech_clips() -> Vec<String> {
    clip_names()
        .into_iter()
        .filter(|c| c != "silence")
        .collect()
}

/// The held-out clips (test-data/clips-extra/clips.json), never used to choose the heads.
pub fn extra_clips() -> Vec<String> {
    read_json(&repo_root().join("test-data/clips-extra/clips.json"))
        .as_array()
        .expect("a clip list")
        .iter()
        .map(|c| c["name"].as_str().expect("a name").to_string())
        .collect()
}

/// `dir` is `clips` or `clips-extra`.
pub fn wav_in(dir: &str, clip: &str) -> Vec<u8> {
    let path = repo_root()
        .join("test-data")
        .join(dir)
        .join(format!("{clip}.wav"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub fn wav(clip: &str) -> Vec<u8> {
    wav_in("clips", clip)
}

pub fn timed() -> Options {
    Options {
        word_timestamps: true,
        ..Options::default()
    }
}

/// One engine word: the text, start, end and probability.
#[derive(Debug, Clone)]
pub struct EngineWord {
    pub word: String,
    pub start: f32,
    pub end: f32,
    pub probability: f32,
}

/// The engine's text and words for `clip` in `dir` (`engine` or `engine-extra`).
pub fn engine(dir: &str, clip: &str) -> (String, Vec<EngineWord>) {
    let v = read_json(&repo_root().join(format!("test-data/{dir}/{clip}.json")));
    let f = |w: &serde_json::Value, k: &str| w[k].as_f64().expect("a number") as f32;
    let words = v["words"]
        .as_array()
        .expect("words")
        .iter()
        .map(|w| EngineWord {
            word: w["word"].as_str().expect("a word").to_string(),
            start: f(w, "start"),
            end: f(w, "end"),
            probability: f(w, "probability"),
        })
        .collect();
    (v["text"].as_str().expect("text").to_string(), words)
}
