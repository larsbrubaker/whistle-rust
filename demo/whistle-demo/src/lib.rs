//! # whistle-demo
//!
//! The UI of the whistle-rust browser demo, built with agg-gui: a Record/Stop button, the live
//! transcript (committed text, then the pending tail in a lighter style), an options row
//! (language, beams, keywords, word timestamps), a status line (model download, loading,
//! listening, each update's compute time and real-time factor), transcribe-a-file (pick, or
//! drop a WAV anywhere on the page) and the model's license note.
//!
//! The UI never transcribes: it calls the shell through [`DemoHost`] (start or stop the
//! microphone, pick or send a file, open a link), and the shell reports back into
//! [`DemoState`]. demo/whistle-demo-web implements the host in the browser, where the model
//! runs in a Web Worker so the UI never blocks; demo/whistle-demo-native implements it on the
//! desktop (agg-gui-shell), with the model on an engine thread. [`Platform`] holds the few
//! words that name where the demo runs.

mod drop_target;
mod state;
mod ui;

use std::rc::Rc;
use std::sync::Arc;

use agg_gui::fonts::standard_ui_font;
use agg_gui::text::Font;
use agg_gui::{font_settings, set_device_scale, App};

pub use state::{DemoState, Phase, TimedWord, UiOptions, BEAM_CHOICES, LANGUAGE_CHOICES};

/// The page's title.
pub const APP_TITLE: &str = "Whistle: speech to text in your browser";

/// The desktop window's title.
pub const DESKTOP_TITLE: &str = "Whistle: speech to text on your desktop";

/// The words that differ between the shells: the title, the line under it, and where a file
/// can be dropped.
#[derive(Debug, Clone, PartialEq)]
pub struct Platform {
    pub title: &'static str,
    pub blurb: &'static str,
    /// Beside the file button: where a WAV can be dropped.
    pub drop_hint: &'static str,
}

impl Platform {
    /// The browser demo's words.
    pub fn browser() -> Platform {
        Platform {
            title: APP_TITLE,
            blurb: "Live speech-to-text with Whistle, running entirely in this page \
                    (pure Rust, WebAssembly, one thread). Nothing you say leaves your device.",
            drop_hint: "or drop one on the page",
        }
    }

    /// The desktop demo's words.
    pub fn desktop() -> Platform {
        Platform {
            title: DESKTOP_TITLE,
            blurb: "Live speech-to-text with Whistle, running entirely on this computer \
                    (pure Rust). Nothing you say leaves your device.",
            drop_hint: "or drop one on the window",
        }
    }
}

/// Where the model's license note links.
pub const MODEL_LICENSE_URL: &str = "https://huggingface.co/Cactus-Compute/whistle";

/// Where the demo's source lives.
pub const SOURCE_URL: &str = "https://github.com/larsbrubaker/whistle-rust";

/// What the UI asks of its shell. Every call returns at once; results arrive through
/// [`DemoState`].
pub trait DemoHost {
    /// Ask for the microphone and start streaming with `options`.
    fn start_recording(&self, options: &UiOptions);
    /// Stop the microphone and finish the stream (the final transcript follows).
    fn stop_recording(&self);
    /// Let the user pick a WAV file and transcribe it with `options`.
    fn pick_file(&self, options: &UiOptions);
    /// Transcribe a dropped file with `options`.
    fn transcribe_file(&self, name: &str, bytes: Arc<[u8]>, options: &UiOptions);
    /// Open `url` in a new tab.
    fn open_url(&self, url: &str);
}

/// Install the UI font for a display at `device_scale` and return it.
pub fn install_fonts(device_scale: f64) -> Arc<Font> {
    set_device_scale(device_scale);
    let font = Arc::new(standard_ui_font());
    font_settings::set_system_font(Some(Arc::clone(&font)));
    font
}

/// The browser demo's app: the UI over `state`, calling `host`.
pub fn build_app(state: DemoState, host: Rc<dyn DemoHost>, font: Arc<Font>) -> App {
    build_app_on(state, host, font, &Platform::browser())
}

/// The demo's app with `platform`'s words.
pub fn build_app_on(
    state: DemoState,
    host: Rc<dyn DemoHost>,
    font: Arc<Font>,
    platform: &Platform,
) -> App {
    App::new(ui::build_root(state, host, font, platform))
}
