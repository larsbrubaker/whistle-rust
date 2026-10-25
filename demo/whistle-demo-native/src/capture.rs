//! The seam a live audio source plugs into: [`Capture`] starts delivering mono chunks to a
//! [`Sink`] on a thread of its own and stops on request. The bridge (`bridge.rs`) forwards each
//! chunk to the engine thread (`engine.rs`) as a stream push, as the browser's microphone feeds
//! its worker. The system microphone is `MicCapture` (`mic.rs`, over agg-mic); where agg-mic has
//! no microphone support the desktop demo has no capture and shows Record unavailable.
//!
//! [`WavCapture`] plays a WAV file through the same seam, in real time or as fast as it can: the
//! tests stream clips with it, and `--mic-from-wav` runs the live path without a microphone.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// What a capture tells its sink.
#[derive(Debug, Clone, PartialEq)]
pub enum CaptureEvent {
    /// Mono samples at `rate` Hz, in order.
    Audio { samples: Vec<f32>, rate: u32 },
    /// The source ended on its own (a file ran out, a device went away; the text says what
    /// failed, if anything). No audio follows.
    Ended(Option<String>),
}

/// Where a capture delivers its events (called on the capture's thread).
pub type Sink = Box<dyn FnMut(CaptureEvent) + Send>;

/// A live audio source.
pub trait Capture {
    /// Start delivering audio to `sink`. An error is a message a user can act on.
    fn start(&mut self, sink: Sink) -> Result<(), String>;
    /// Stop. Every chunk delivered is in the sink before this returns, and none follows, so a
    /// caller that finishes the stream next has all of the audio.
    fn stop(&mut self);
}

/// Audio per chunk: 100 ms, about what the browser's worklet hands over.
const CHUNK_SECONDS: f64 = 0.1;

/// A WAV file played through the [`Capture`] seam.
pub struct WavCapture {
    samples: Arc<[f32]>,
    rate: u32,
    /// Pace chunks at the audio's own speed (a stand-in microphone), or deliver them at once.
    realtime: bool,
    running: Option<(Arc<AtomicBool>, JoinHandle<()>)>,
}

impl WavCapture {
    /// The WAV in `bytes`, mixed to mono by the library's reader.
    pub fn from_wav(bytes: &[u8], realtime: bool) -> Result<WavCapture, String> {
        let audio =
            whistle::audio::read_wav(bytes).map_err(|e| format!("not a usable WAV: {e}"))?;
        Ok(WavCapture {
            samples: audio.samples.into(),
            rate: audio.sample_rate,
            realtime,
            running: None,
        })
    }

    /// Samples per chunk at the file's rate.
    fn chunk_len(&self) -> usize {
        // Rates are at most a few hundred kHz, so the product fits.
        ((self.rate as f64 * CHUNK_SECONDS) as usize).max(1)
    }
}

impl Capture for WavCapture {
    fn start(&mut self, mut sink: Sink) -> Result<(), String> {
        self.stop();
        let stop = Arc::new(AtomicBool::new(false));
        let (samples, rate, realtime) = (Arc::clone(&self.samples), self.rate, self.realtime);
        let chunk = self.chunk_len();
        let stopped = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("whistle-demo-wav-capture".into())
            .spawn(move || {
                let began = Instant::now();
                for (i, part) in samples.chunks(chunk).enumerate() {
                    if realtime {
                        // Deliver each chunk once its audio has "happened", as a device would.
                        let due = Duration::from_secs_f64((i + 1) as f64 * CHUNK_SECONDS);
                        if let Some(wait) = due.checked_sub(began.elapsed()) {
                            std::thread::sleep(wait);
                        }
                    }
                    if stopped.load(Ordering::SeqCst) {
                        return;
                    }
                    sink(CaptureEvent::Audio {
                        samples: part.to_vec(),
                        rate,
                    });
                }
                if !stopped.load(Ordering::SeqCst) {
                    sink(CaptureEvent::Ended(None));
                }
            })
            .map_err(|e| format!("could not start the capture thread: {e}"))?;
        self.running = Some((stop, thread));
        Ok(())
    }

    fn stop(&mut self) {
        if let Some((stop, thread)) = self.running.take() {
            stop.store(true, Ordering::SeqCst);
            // A panicked capture thread has delivered all it will; nothing more to wait for.
            let _ = thread.join();
        }
    }
}

impl Drop for WavCapture {
    fn drop(&mut self) {
        self.stop();
    }
}
