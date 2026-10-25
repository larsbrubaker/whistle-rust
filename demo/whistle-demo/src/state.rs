//! The demo's state: where the model is (downloading, loading, ready), whether the microphone
//! is live, the transcript so far (committed text and the pending tail), the last update's
//! timing, and the options the user picked. The shell (whistle-demo-web) feeds it what the
//! transcription worker reports; the UI (`ui.rs`) reads it and rebuilds when [`DemoState::version`]
//! moves. Plain Rust, so the formatting is tested natively.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// The languages the options row offers: "auto" (detect), then the model's seven.
pub const LANGUAGE_CHOICES: [&str; 8] = ["auto", "en", "de", "fr", "es", "it", "nl", "pl"];

/// The beam counts the options row offers.
pub const BEAM_CHOICES: [usize; 2] = [1, 3];

/// What the user picked in the options row (the defaults: auto language, one beam, no
/// keywords, no word timestamps).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UiOptions {
    /// Index into [`LANGUAGE_CHOICES`]; 0 is auto-detect.
    pub language: usize,
    /// Index into [`BEAM_CHOICES`]; 0 (one beam) is the live preset.
    pub beams: usize,
    /// Comma-separated words and phrases to favour.
    pub keywords: String,
    /// Word timestamps on the final transcript.
    pub word_timestamps: bool,
}

impl UiOptions {
    /// The forced language code, or `None` for auto-detect.
    pub fn language_code(&self) -> Option<&'static str> {
        match self.language {
            0 => None,
            i => LANGUAGE_CHOICES.get(i).copied(),
        }
    }

    /// The beam count.
    pub fn beam_count(&self) -> usize {
        BEAM_CHOICES.get(self.beams).copied().unwrap_or(1)
    }

    /// The keywords, split on commas, trimmed, blanks dropped.
    pub fn keyword_list(&self) -> Vec<String> {
        self.keywords
            .split(',')
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .map(str::to_string)
            .collect()
    }
}

/// Where the demo is.
#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    /// Fetching `whistle.cact`: bytes so far and the expected total.
    Downloading { loaded: u64, total: u64 },
    /// Verifying the download and building the model in the worker.
    Loading,
    /// The model is ready and nothing is running.
    Ready,
    /// Asking the browser for the microphone.
    RequestingMic,
    /// The microphone is live and audio streams to the worker.
    Listening,
    /// The stream is finishing, or a file is being transcribed.
    Transcribing,
    /// Something failed; the text says what, in words a user can act on.
    Error(String),
}

/// One word of the final transcript with its time.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedWord {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug)]
struct Inner {
    phase: Phase,
    committed: String,
    pending: String,
    language: Option<String>,
    /// Compute time of the last update, ms, and the audio it covered, s.
    last_update: Option<(f64, f64)>,
    words: Vec<TimedWord>,
    options: UiOptions,
    model_source: String,
    /// Set once the worker reported the model ready.
    model_loaded: bool,
}

/// Shared handle to the demo's state (cheap to clone; single-threaded, like the UI).
#[derive(Clone)]
pub struct DemoState {
    inner: Rc<RefCell<Inner>>,
    version: Rc<Cell<u64>>,
}

impl Default for DemoState {
    fn default() -> DemoState {
        DemoState::new()
    }
}

impl DemoState {
    pub fn new() -> DemoState {
        DemoState {
            inner: Rc::new(RefCell::new(Inner {
                phase: Phase::Downloading {
                    loaded: 0,
                    total: 0,
                },
                committed: String::new(),
                pending: String::new(),
                language: None,
                last_update: None,
                words: Vec::new(),
                options: UiOptions::default(),
                model_source: String::new(),
                model_loaded: false,
            })),
            version: Rc::new(Cell::new(0)),
        }
    }

    /// Bumped on every change the UI shows; the UI's `Rebuilder`s key on it.
    pub fn version(&self) -> u64 {
        self.version.get()
    }

    fn change(&self, f: impl FnOnce(&mut Inner)) {
        f(&mut self.inner.borrow_mut());
        self.version.set(self.version.get().wrapping_add(1));
    }

    pub fn phase(&self) -> Phase {
        self.inner.borrow().phase.clone()
    }

    pub fn options(&self) -> UiOptions {
        self.inner.borrow().options.clone()
    }

    /// Option edits do not bump the version: the widgets that made them already show them, and
    /// a rebuild would take the keyboard focus from the keywords field.
    pub fn edit_options(&self, f: impl FnOnce(&mut UiOptions)) {
        f(&mut self.inner.borrow_mut().options);
    }

    pub fn set_download(&self, loaded: u64, total: u64, source: &str) {
        self.change(|s| {
            s.phase = Phase::Downloading { loaded, total };
            s.model_source = source.to_string();
        });
    }

    pub fn set_phase(&self, phase: Phase) {
        self.change(|s| {
            s.model_loaded |= phase == Phase::Ready;
            s.phase = phase;
        });
    }

    /// Where the model came from (the URL, or "cache").
    pub fn model_source(&self) -> String {
        self.inner.borrow().model_source.clone()
    }

    /// The microphone started: a fresh transcript.
    pub fn start_listening(&self) {
        self.change(|s| {
            s.phase = Phase::Listening;
            s.committed.clear();
            s.pending.clear();
            s.language = None;
            s.last_update = None;
            s.words.clear();
        });
    }

    /// A file started: a fresh transcript.
    pub fn start_file(&self, name: &str) {
        self.change(|s| {
            s.phase = Phase::Transcribing;
            s.committed.clear();
            s.pending = format!("Transcribing {name}\u{2026}");
            s.language = None;
            s.last_update = None;
            s.words.clear();
        });
    }

    /// One stream update: `text` is newly committed (`continues`: appended with no space),
    /// `pending` replaces the tail; `ms` of compute for `audio_s` seconds of new audio. A tail
    /// that continues the last committed word (`pending.1`: "0 mm" after a committed "2") is
    /// shown with that word's start, "20 mm": shown alone it would read as a word "0".
    pub fn apply_update(
        &self,
        text: &str,
        continues: bool,
        pending: (&str, bool),
        language: Option<String>,
        ms: f64,
        audio_s: f64,
    ) {
        self.change(|s| {
            if !text.is_empty() {
                if !s.committed.is_empty() && !continues {
                    s.committed.push(' ');
                }
                s.committed.push_str(text);
            }
            let (tail, tail_continues) = pending;
            s.pending = if tail_continues && !tail.is_empty() {
                let head = s.committed.rsplit(' ').next().unwrap_or_default();
                format!("{head}{tail}")
            } else {
                tail.to_string()
            };
            if language.is_some() {
                s.language = language;
            }
            s.last_update = Some((ms, audio_s));
        });
    }

    /// The final transcript (a finished stream's, or a file's): replaces what was shown.
    pub fn set_final(
        &self,
        text: &str,
        language: Option<String>,
        words: Vec<TimedWord>,
        ms: f64,
        audio_s: f64,
    ) {
        self.change(|s| {
            s.phase = Phase::Ready;
            s.committed = text.to_string();
            s.pending.clear();
            s.language = language;
            s.words = words;
            s.last_update = Some((ms, audio_s));
        });
    }

    pub fn committed(&self) -> String {
        self.inner.borrow().committed.clone()
    }

    /// The unconfirmed tail as shown: with the start of the word it continues, if it does.
    pub fn pending(&self) -> String {
        self.inner.borrow().pending.clone()
    }

    /// The word timings of the final transcript, "word [start–end]" each for the final transcript (empty when none).
    pub fn words_line(&self) -> String {
        let s = self.inner.borrow();
        s.words
            .iter()
            .map(|w| format!("{} [{:.2}\u{2013}{:.2}]", w.word, w.start, w.end))
            .collect::<Vec<_>>()
            .join("  ")
    }

    /// Whether the Record button should read "Stop".
    pub fn is_listening(&self) -> bool {
        matches!(
            self.inner.borrow().phase,
            Phase::Listening | Phase::RequestingMic
        )
    }

    /// Whether Record and the file picker can start something now: the model has loaded and
    /// nothing is running (an error after loading leaves the model usable).
    pub fn can_start(&self) -> bool {
        let s = self.inner.borrow();
        s.model_loaded && matches!(s.phase, Phase::Ready | Phase::Error(_))
    }

    /// The status line: model progress, "loading model", "listening", and the last update's
    /// compute time and real-time factor (compute time over audio time; under 1 keeps up).
    pub fn status_line(&self) -> String {
        let s = self.inner.borrow();
        let phase = match &s.phase {
            Phase::Downloading { loaded, total } if *total > 0 => format!(
                "Downloading model {:.1} / {:.1} MB ({:.0}%)",
                *loaded as f64 / 1e6,
                *total as f64 / 1e6,
                *loaded as f64 * 100.0 / *total as f64
            ),
            Phase::Downloading { loaded, .. } => {
                format!("Downloading model {:.1} MB", *loaded as f64 / 1e6)
            }
            Phase::Loading => "Loading model\u{2026}".to_string(),
            Phase::Ready => "Ready".to_string(),
            Phase::RequestingMic => "Asking for the microphone\u{2026}".to_string(),
            Phase::Listening => "Listening".to_string(),
            Phase::Transcribing => "Transcribing\u{2026}".to_string(),
            Phase::Error(e) => format!("Error: {e}"),
        };
        let mut line = phase;
        if let Some(lang) = &s.language {
            line.push_str(&format!(" \u{b7} {lang}"));
        }
        if let Some((ms, audio_s)) = s.last_update {
            line.push_str(&format!(" \u{b7} last update {ms:.0} ms"));
            if audio_s > 0.0 {
                line.push_str(&format!(", RTF {:.2}", ms / 1000.0 / audio_s));
            }
        }
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn updates_append_committed_text_and_replace_the_pending_tail() {
        let state = DemoState::new();
        state.start_listening();
        state.apply_update(
            "Hello",
            false,
            ("wor", false),
            Some("en".into()),
            120.0,
            1.0,
        );
        state.apply_update("world", false, ("", false), None, 80.0, 1.0);
        state.apply_update("s.", true, ("and", false), None, 80.0, 1.0);
        assert_eq!(state.committed(), "Hello worlds.");
        assert_eq!(state.pending(), "and");
    }

    #[test]
    fn a_pending_tail_that_continues_the_last_word_is_shown_with_its_start() {
        // "2" committed, then a pass reading "20 mm": the tail "0 mm" continues "2", so the
        // preview shows "20 mm", never a word "0".
        let state = DemoState::new();
        state.start_listening();
        state.apply_update("Make the cube 2", false, ("0 mm", true), None, 80.0, 1.0);
        assert_eq!(state.committed(), "Make the cube 2");
        assert_eq!(state.pending(), "20 mm");
        // Once the rest commits, the next tail stands on its own.
        state.apply_update("0 mm", true, ("wide", false), None, 80.0, 1.0);
        assert_eq!(state.committed(), "Make the cube 20 mm");
        assert_eq!(state.pending(), "wide");
    }

    #[test]
    fn status_line_shows_progress_then_timing_and_real_time_factor() {
        let state = DemoState::new();
        state.set_download(8_000_000, 16_000_000, "./whistle.cact");
        assert_eq!(state.status_line(), "Downloading model 8.0 / 16.0 MB (50%)");
        state.set_phase(Phase::Ready);
        state.start_listening();
        state.apply_update("Hi", false, ("", false), Some("en".into()), 250.0, 1.0);
        assert_eq!(
            state.status_line(),
            "Listening \u{b7} en \u{b7} last update 250 ms, RTF 0.25"
        );
    }

    #[test]
    fn options_map_to_language_beams_and_keywords() {
        let options = UiOptions {
            language: 2,
            beams: 1,
            keywords: " PETG, , MatterCAD ".into(),
            word_timestamps: true,
        };
        assert_eq!(options.language_code(), Some("de"));
        assert_eq!(options.beam_count(), 3);
        assert_eq!(options.keyword_list(), vec!["PETG", "MatterCAD"]);
        assert_eq!(UiOptions::default().language_code(), None);
        assert_eq!(UiOptions::default().beam_count(), 1);
    }

    #[test]
    fn option_edits_do_not_rebuild_the_ui() {
        let state = DemoState::new();
        let before = state.version();
        state.edit_options(|o| o.keywords = "x".into());
        assert_eq!(state.version(), before);
        state.set_phase(Phase::Ready);
        assert_ne!(state.version(), before);
    }
}
