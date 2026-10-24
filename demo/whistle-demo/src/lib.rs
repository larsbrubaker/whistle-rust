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
//! runs in a Web Worker so the UI never blocks.

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

/// The demo's app: the UI over `state`, calling `host`.
pub fn build_app(state: DemoState, host: Rc<dyn DemoHost>, font: Arc<Font>) -> App {
    App::new(ui::build_root(state, host, font))
}
