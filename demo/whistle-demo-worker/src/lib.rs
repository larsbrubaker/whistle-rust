//! # whistle-demo-worker
//!
//! The browser demo's transcription worker module: [`Engine`] wraps one `Whistle` and at most
//! one live `Stream`, and answers each call with a small JSON string that `demo/web/src/worker.ts`
//! forwards to the page. The worker is one thread, like the phones and slow laptops the demo
//! targets; the page's UI (whistle-demo-web) never runs the model, so it never blocks. Timing
//! is measured by the worker around each call.
//!
//! Natively the crate is empty (a workspace member only so fmt and clippy cover it).

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use whistle::{Options, Stream, StreamUpdate, Transcript, Whistle, Word};

/// The model and the live stream, in the transcription worker.
#[wasm_bindgen]
pub struct Engine {
    /// Leaked once per worker: a `Stream` borrows its `Whistle`, and the worker keeps the
    /// model for its whole life, so a `'static` borrow is the honest lifetime.
    whistle: &'static Whistle,
    stream: Option<Stream<'static>>,
    /// Every word the stream committed, for the final transcript's timings.
    words: Vec<Word>,
    /// Every word's text the stream committed, joined as the UI joins updates.
    text: String,
    /// The stream's options, kept to know whether to report timings.
    word_timestamps: bool,
}

fn options(language: &str, beams: usize, keywords: Vec<String>, word_timestamps: bool) -> Options {
    Options {
        language: (!language.is_empty()).then(|| language.to_string()),
        beams: beams.max(1),
        keywords,
        word_timestamps,
        ..Options::streaming()
    }
}

fn js_err(err: whistle::Error) -> JsValue {
    JsValue::from_str(&err.to_string())
}

#[wasm_bindgen]
impl Engine {
    /// Build the model from the bytes of `whistle.cact` (already size- and SHA-256-checked).
    #[wasm_bindgen(constructor)]
    pub fn new(model: &[u8]) -> Result<Engine, JsValue> {
        let whistle = Whistle::from_cact(model).map_err(js_err)?;
        Ok(Engine {
            whistle: Box::leak(Box::new(whistle)),
            stream: None,
            words: Vec::new(),
            text: String::new(),
            word_timestamps: false,
        })
    }

    /// Open a live stream (`Options::streaming()` with these choices; "" is auto language).
    pub fn start(
        &mut self,
        language: &str,
        beams: usize,
        keywords: Vec<String>,
        word_timestamps: bool,
    ) -> Result<(), JsValue> {
        let options = options(language, beams, keywords, word_timestamps);
        self.stream = Some(self.whistle.stream(&options).map_err(js_err)?);
        self.words.clear();
        self.text.clear();
        self.word_timestamps = word_timestamps;
        Ok(())
    }

    /// Push mono samples at `rate` Hz (resampled to 16 kHz when needed); the update as JSON
    /// `{"text", "continues", "pending", "pendingContinues", "language"}`.
    pub fn push(&mut self, samples: &[f32], rate: u32) -> Result<String, JsValue> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsValue::from_str("no stream is open"))?;
        let update = if rate == 16_000 {
            stream.push(samples)
        } else {
            stream.push(&whistle::audio::resample(samples, rate, 16_000))
        }
        .map_err(js_err)?;
        self.record(&update);
        Ok(update_json(&update))
    }

    /// Finish the stream: the last update as JSON, plus `"words"` (every committed word with
    /// its times, when word timestamps were asked for) and `"full"` (the whole text).
    pub fn finish(&mut self) -> Result<String, JsValue> {
        let stream = self
            .stream
            .take()
            .ok_or_else(|| JsValue::from_str("no stream is open"))?;
        let update = stream.finish().map_err(js_err)?;
        self.record(&update);
        let words = if self.word_timestamps {
            words_json(&self.words)
        } else {
            "[]".to_string()
        };
        let json = update_json(&update);
        Ok(format!(
            "{},\"words\":{words},\"full\":{}}}",
            &json[..json.len() - 1],
            json_string(&self.text)
        ))
    }

    /// Keep an update's words and text for the final transcript.
    fn record(&mut self, update: &StreamUpdate) {
        self.words.extend(update.committed.iter().cloned());
        if !update.text.is_empty() {
            if !self.text.is_empty() && !update.continues {
                self.text.push(' ');
            }
            self.text.push_str(&update.text);
        }
    }

    /// Transcribe a whole WAV file; the transcript as JSON `{"text", "language", "words"}`.
    pub fn transcribe_wav(
        &self,
        wav: &[u8],
        language: &str,
        beams: usize,
        keywords: Vec<String>,
        word_timestamps: bool,
    ) -> Result<String, JsValue> {
        // The batch decode, with the beams the user picked.
        let options = options(language, beams, keywords, word_timestamps);
        let transcript = self.whistle.transcribe_wav(wav, &options).map_err(js_err)?;
        Ok(transcript_json(&transcript))
    }

    /// Seconds of audio a WAV file holds at 16 kHz after decoding (for the real-time factor).
    pub fn wav_seconds(wav: &[u8]) -> f64 {
        whistle::audio::read_wav(wav)
            .map(|audio| audio.samples.len() as f64 / audio.sample_rate.max(1) as f64)
            .unwrap_or(0.0)
    }
}

/// `whistle.cact`'s pinned URL (Hugging Face), size and SHA-256, for the page's download.
#[wasm_bindgen]
pub fn model_url() -> String {
    whistle::model_file::MODEL_URL.to_string()
}

#[wasm_bindgen]
pub fn model_sha256() -> String {
    whistle::model_file::MODEL_SHA256.to_string()
}

#[wasm_bindgen]
pub fn model_bytes() -> f64 {
    // Exact: the size is far below 2^53.
    whistle::model_file::MODEL_BYTES as f64
}

/// `s` as a JSON string literal.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn language_json(language: &Option<String>) -> String {
    language
        .as_deref()
        .map(json_string)
        .unwrap_or_else(|| "null".to_string())
}

fn words_json(words: &[Word]) -> String {
    let items: Vec<String> = words
        .iter()
        .map(|w| {
            format!(
                "{{\"word\":{},\"start\":{},\"end\":{}}}",
                json_string(&w.word),
                w.start,
                w.end
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

fn update_json(update: &StreamUpdate) -> String {
    format!(
        "{{\"text\":{},\"continues\":{},\"pending\":{},\"pendingContinues\":{},\"language\":{}}}",
        json_string(&update.text),
        update.continues,
        json_string(&update.pending),
        update.pending_continues,
        language_json(&update.language)
    )
}

fn transcript_json(transcript: &Transcript) -> String {
    format!(
        "{{\"text\":{},\"language\":{},\"words\":{}}}",
        json_string(&transcript.text),
        language_json(&transcript.language),
        words_json(&transcript.words)
    )
}
