//! The system microphone as a [`Capture`] (`capture.rs`): [`MicCapture`] records from agg-mic's
//! [`Microphone`] (AVFoundation on macOS, WASAPI on Windows, the PulseAudio protocol on Linux,
//! all pure Rust) and hands each chunk to the sink at the device's own rate; the engine thread
//! (`engine.rs`) resamples the stream to the model's 16 kHz, as it does for any capture.
//!
//! agg-mic's start blocks until audio flows, which on macOS includes the permission prompt the
//! first time, for as long as the user leaves it up. The bridge (`bridge.rs`) starts captures
//! on the UI thread, so the start runs on a thread of its own and [`Capture::start`] returns at
//! once; a start that fails (permission refused, no input device) and a recording that ends on
//! its own (unplugged) reach the sink as [`CaptureEvent::Ended`] with agg-mic's message, ready
//! to show. A stop is agg-mic's own, which is quick and delivers the device's last audio inside
//! it, so the bridge's Finish follows every chunk. A stop cannot take down a permission prompt,
//! so a start waiting on one outlives its stop; until it ends, a new start is refused with
//! [`STILL_STARTING_MESSAGE`], which says what to do, rather than agg-mic's "already
//! recording".
//!
//! **macOS permission.** A bare `cargo run` binary is not an app bundle, so macOS attributes
//! its microphone use to the app that launched it (Terminal, iTerm, VS Code or another IDE):
//! the first Record asks for, or uses, that app's permission, which every program it starts then
//! shares, and turning it off is under System Settings > Privacy & Security > Microphone for that
//! app. If the launching app's Info.plist lacks `NSMicrophoneUsageDescription`, macOS may end
//! the process at Record instead of prompting. A bundled build would need that key in its own
//! Info.plist and, under the hardened runtime, the `com.apple.security.device.audio-input`
//! entitlement (without it the process gets silence and no prompt).

use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;

use agg_mic::{MicError, Microphone, UNSUPPORTED_MESSAGE};

use crate::capture::{Capture, CaptureEvent, Sink};

/// What a Record says while the last one's start still waits: on macOS for its permission
/// prompt, which a Stop cannot take down (agg-mic would only call it already recording).
pub const STILL_STARTING_MESSAGE: &str = "The microphone is still starting from the last Record. Answer the microphone permission prompt if one is open, then press Record again.";

/// The running recording's sink, shared by agg-mic's handlers and the start thread; `None` once
/// the recording stopped or ended, so nothing reaches the sink after that.
type SharedSink = Arc<Mutex<Option<Sink>>>;

/// Hands `event` to the sink if the recording is still on; `last` ends it (no event follows).
fn tell(sink: &SharedSink, event: CaptureEvent, last: bool) {
    let mut slot = sink.lock().unwrap_or_else(PoisonError::into_inner);
    if last {
        if let Some(mut sink) = slot.take() {
            sink(event);
        }
    } else if let Some(sink) = slot.as_mut() {
        sink(event);
    }
}

/// See the module docs.
pub struct MicCapture {
    microphone: Arc<Microphone>,
    sink: Option<SharedSink>,
    /// The thread of the last start: it ends once agg-mic's start returns.
    starting: Option<JoinHandle<()>>,
}

impl MicCapture {
    /// A capture over `microphone`, or, where agg-mic has no microphone support, why Record is
    /// unavailable (agg-mic's message). Neither asks for permission: only a start does.
    pub fn new(microphone: Microphone) -> Result<MicCapture, String> {
        if !microphone.is_available() {
            return Err(UNSUPPORTED_MESSAGE.to_owned());
        }
        Ok(MicCapture {
            microphone: Arc::new(microphone),
            sink: None,
            starting: None,
        })
    }
}

impl Capture for MicCapture {
    fn start(&mut self, sink: Sink) -> Result<(), String> {
        self.stop();
        if self.starting.as_ref().is_some_and(|t| !t.is_finished()) {
            return Err(STILL_STARTING_MESSAGE.to_owned());
        }
        let sink: SharedSink = Arc::new(Mutex::new(Some(sink)));
        let (chunks, failure, refusal) = (Arc::clone(&sink), Arc::clone(&sink), Arc::clone(&sink));
        let microphone = Arc::clone(&self.microphone);
        let thread = std::thread::Builder::new()
            .name("whistle-demo-mic-start".into())
            .spawn(move || {
                let started = microphone.start(
                    Box::new(move |samples: &[f32], rate: u32| {
                        let samples = samples.to_vec();
                        tell(&chunks, CaptureEvent::Audio { samples, rate }, false);
                    }),
                    Box::new(move |error: MicError| {
                        tell(&failure, CaptureEvent::Ended(Some(error.to_string())), true);
                    }),
                );
                match started {
                    // A stop while the start waited (on the prompt): the stop already took the
                    // sink, and there is nothing to report.
                    Ok(()) | Err(MicError::Stopped) => {}
                    Err(error) => {
                        tell(&refusal, CaptureEvent::Ended(Some(error.to_string())), true);
                    }
                }
            })
            .map_err(|e| format!("could not start the microphone thread: {e}"))?;
        self.sink = Some(sink);
        self.starting = Some(thread);
        Ok(())
    }

    fn stop(&mut self) {
        // agg-mic's stop delivers every chunk captured before it returns (through the sink),
        // and none after; a start still waiting on the prompt ends as stopped. Taking the sink
        // then keeps a start that failed meanwhile from reporting after this returns.
        self.microphone.stop();
        if let Some(sink) = self.sink.take() {
            drop(sink.lock().unwrap_or_else(PoisonError::into_inner).take());
        }
    }
}

impl Drop for MicCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
#[path = "mic_tests.rs"]
mod tests;
