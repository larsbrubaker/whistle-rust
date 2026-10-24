// Shared state for the transcription test area: the model bound once from models/whistle.cact
// (both as a `Whistle` and as bare `Weights` for the decoder-level tests), the speech clip
// names, and each clip's WAV bytes from test-data/clips/.

use std::sync::OnceLock;

use whistle::cact::Archive;
use whistle::model::Weights;
use whistle::Whistle;

use crate::support::archive::model_bytes;
use crate::support::{clip_names, repo_root};

pub fn whistle() -> &'static Whistle {
    static WHISTLE: OnceLock<Whistle> = OnceLock::new();
    WHISTLE.get_or_init(|| Whistle::from_cact(&model_bytes()).expect("whistle.cact loads"))
}

pub fn weights() -> &'static Weights {
    static WEIGHTS: OnceLock<Weights> = OnceLock::new();
    WEIGHTS.get_or_init(|| {
        let bytes = model_bytes();
        let archive = Archive::parse(&bytes).expect("whistle.cact parses");
        Weights::from_archive(&archive).expect("whistle.cact binds")
    })
}

/// Every clip but `silence`.
pub fn speech_clips() -> Vec<String> {
    clip_names()
        .into_iter()
        .filter(|c| c != "silence")
        .collect()
}

pub fn wav(clip: &str) -> Vec<u8> {
    let path = repo_root()
        .join("test-data/clips")
        .join(format!("{clip}.wav"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The manifest's text tokens: `tokens` without BOS, the language tag and EOS.
pub fn text_tokens(tokens: &[u32]) -> Vec<u32> {
    tokens[2..tokens.len() - 1].to_vec()
}
