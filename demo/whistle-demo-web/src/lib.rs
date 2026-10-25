//! # whistle-demo-web
//!
//! The browser demo's UI module, on the page's main thread (`demo/web/src/main.ts`): [`start_ui`]
//! runs the whistle-demo UI in agg-gui-web-shell on the page's canvas. The UI calls the page
//! through `globalThis.whistleHost` (record, stop, pick a file, transcribe a dropped file, open
//! a link); the page reports what the transcription worker (demo/whistle-demo-worker) says
//! through the `ui_*` exports, which update the [`DemoState`] and wake the frame loop.
//!
//! Natively the crate is empty (a workspace member only so fmt and clippy cover it).

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use agg_gui_web_shell::{mark_dirty, start, NoHost, RedrawPolicy, WebShellConfig};
use wasm_bindgen::prelude::*;
use whistle_demo::{build_app, install_fonts, DemoHost, DemoState, Phase, TimedWord, UiOptions};

/// `id` of the canvas in `demo/web/index.html`.
const CANVAS_ID: &str = "whistle-canvas";

#[wasm_bindgen(inline_js = "export function host_call(name, args) { \
    const h = globalThis.whistleHost; \
    if (h && typeof h[name] === 'function') { h[name](...args); } }")]
extern "C" {
    /// `globalThis.whistleHost[name](...args)`, when the page installed it.
    fn host_call(name: &str, args: js_sys::Array);
}

thread_local! {
    static STATE: RefCell<Option<DemoState>> = const { RefCell::new(None) };
}

/// The options as the page's host functions take them: language ("" for auto), beams,
/// keywords (one per array entry), word timestamps.
fn options_args(options: &UiOptions) -> js_sys::Array {
    let keywords = js_sys::Array::new();
    for k in options.keyword_list() {
        keywords.push(&JsValue::from_str(&k));
    }
    let args = js_sys::Array::new();
    args.push(&JsValue::from_str(options.language_code().unwrap_or("")));
    args.push(&JsValue::from_f64(options.beam_count() as f64));
    args.push(&keywords);
    args.push(&JsValue::from_bool(options.word_timestamps));
    args
}

/// The UI's calls, forwarded to the page.
struct WebHost;

impl DemoHost for WebHost {
    fn start_recording(&self, options: &UiOptions) {
        with_state(|s| s.set_phase(Phase::RequestingMic));
        host_call("startRecording", options_args(options));
    }
    fn stop_recording(&self) {
        with_state(|s| s.set_phase(Phase::Transcribing));
        host_call("stopRecording", js_sys::Array::new());
    }
    fn pick_file(&self, options: &UiOptions) {
        host_call("pickFile", options_args(options));
    }
    fn transcribe_file(&self, name: &str, bytes: Arc<[u8]>, options: &UiOptions) {
        with_state(|s| s.start_file(name));
        let args = options_args(options);
        args.unshift(&js_sys::Uint8Array::from(&bytes[..]));
        args.unshift(&JsValue::from_str(name));
        host_call("transcribeFile", args);
    }
    fn open_url(&self, url: &str) {
        let args = js_sys::Array::new();
        args.push(&JsValue::from_str(url));
        host_call("openUrl", args);
    }
}

/// Run `f` on the state, then repaint (the `Rebuilder`s rebuild on the next layout).
fn with_state(f: impl FnOnce(&DemoState)) {
    STATE.with(|s| {
        if let Some(state) = s.borrow().as_ref() {
            f(state);
        }
    });
    agg_gui_web_shell::agg_gui::animation::request_layout();
    mark_dirty();
}

/// Start the UI on the page's canvas.
#[wasm_bindgen]
pub fn start_ui() -> Result<(), JsValue> {
    let config = WebShellConfig::new(CANVAS_ID)
        .with_app_name("Whistle demo")
        .with_device_label("whistle-demo-web")
        .with_redraw_policy(RedrawPolicy::Reactive);
    start(config, |init| {
        let font = install_fonts(init.device_scale());
        let state = DemoState::new();
        STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
        let host: Rc<dyn DemoHost> = Rc::new(WebHost);
        Ok((build_app(state, host, font), NoHost))
    })
    .map_err(|err| JsValue::from_str(&err.to_string()))
}

/// Model download progress: `loaded` of `total` bytes (0 when unknown) from `source`.
#[wasm_bindgen]
pub fn ui_download(loaded: f64, total: f64, source: &str) {
    // f64 → u64 saturates; byte counts are far below 2^53.
    with_state(|s| s.set_download(loaded as u64, total as u64, source));
}

/// The phase by name: "loading", "ready", "listening", or "error" with `message`.
#[wasm_bindgen]
pub fn ui_phase(name: &str, message: &str) {
    with_state(|s| match name {
        "loading" => s.set_phase(Phase::Loading),
        "listening" => s.start_listening(),
        "error" => s.set_phase(Phase::Error(message.to_string())),
        _ => s.set_phase(Phase::Ready),
    });
}

/// One stream update (`pending_continues`: the tail continues the last committed word); `ms`
/// of compute for `audio_s` seconds of new audio.
#[wasm_bindgen]
pub fn ui_update(
    text: &str,
    continues: bool,
    pending: &str,
    pending_continues: bool,
    language: Option<String>,
    ms: f64,
    audio_s: f64,
) {
    with_state(|s| {
        s.apply_update(
            text,
            continues,
            (pending, pending_continues),
            language,
            ms,
            audio_s,
        )
    });
}

/// The final transcript: `text`, its language, the words ("\n"-separated) with their start
/// and end times (`times` is start, end per word; empty when timestamps are off).
#[wasm_bindgen]
pub fn ui_final(
    text: &str,
    language: Option<String>,
    words: &str,
    times: &[f32],
    ms: f64,
    audio_s: f64,
) {
    let words: Vec<TimedWord> = words
        .split('\n')
        .filter(|w| !w.is_empty())
        .zip(times.as_chunks::<2>().0)
        .map(|(word, &[start, end])| TimedWord {
            word: word.to_string(),
            start,
            end,
        })
        .collect();
    with_state(|s| s.set_final(text, language, words, ms, audio_s));
}

/// Transcribe `bytes` as if the file `name` were dropped on the page, with the current
/// options (page automation; a real drop takes the same path).
#[wasm_bindgen]
pub fn ui_drop_file(name: &str, bytes: &[u8]) {
    let Some(state) = STATE.with(|s| s.borrow().clone()) else {
        return;
    };
    if state.can_start() {
        WebHost.transcribe_file(name, bytes.into(), &state.options());
    }
}

/// `s` as a JSON string literal.
fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The UI's state for page automation: `{"status": .., "committed": .., "pending": ..}`.
#[wasm_bindgen]
pub fn ui_snapshot() -> String {
    STATE.with(|s| match s.borrow().as_ref() {
        Some(state) => format!(
            "{{\"status\":{},\"committed\":{},\"pending\":{},\"words\":{}}}",
            json_string(&state.status_line()),
            json_string(&state.committed()),
            json_string(&state.pending()),
            json_string(&state.words_line()),
        ),
        None => "{\"status\":\"starting\"}".to_string(),
    })
}
