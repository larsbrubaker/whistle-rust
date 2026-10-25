//! The engine thread: the desktop counterpart of the browser demo's Web Worker
//! (`demo/web/src/worker.ts` over `whistle-demo-worker`'s `Engine`). One thread owns the model
//! and at most one live stream, answers [`ToEngine`] messages in order with [`FromEngine`]
//! replies, and times each call, so the UI thread never runs the model. The whistle library's
//! own worker threads (`whistle::set_threads`, chosen in `main.rs`) split each pass's large
//! projections.
//!
//! Every recording and file is a session, numbered by the bridge (`bridge.rs`); its messages and
//! replies carry the number, so audio queued for a stream that has ended is dropped here and a
//! late reply to an old session is dropped by the bridge. Audio at the device's rate goes
//! through one `whistle::audio::StreamResampler` per stream, so the pushes come out as the
//! whole recording would, with no seam at their boundaries.
//!
//! The thread starts by loading the model (`model.rs`: a path, the checkout's copy, the cache or
//! a download), reporting progress, then serves messages until its inbox closes.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use whistle::audio::StreamResampler;
use whistle::{Options, Stream, StreamUpdate, Whistle, Word};
use whistle_demo::{TimedWord, UiOptions};

use crate::model::ModelSource;

/// A recording's or a file's number (the bridge counts them).
pub type Session = u64;

/// The model's sample rate.
const MODEL_RATE: u32 = whistle::audio::SAMPLE_RATE;

/// What the UI asks of the engine.
#[derive(Debug, Clone)]
pub enum ToEngine {
    /// Open a live stream with these options (closing any other).
    Start {
        session: Session,
        options: UiOptions,
    },
    /// Mono samples at `rate` Hz for the session's stream.
    Push {
        session: Session,
        samples: Vec<f32>,
        rate: u32,
    },
    /// Finish the session's stream: the final transcript follows.
    Finish { session: Session },
    /// Drop the session's stream without a transcript (the microphone failed).
    Cancel { session: Session },
    /// Transcribe a whole WAV file.
    File {
        session: Session,
        name: String,
        bytes: Arc<[u8]>,
        options: UiOptions,
    },
}

/// What the engine reports, as the browser worker's messages.
#[derive(Debug, Clone, PartialEq)]
pub enum FromEngine {
    /// Download progress: `loaded` of `total` bytes from `source`.
    Progress {
        loaded: u64,
        total: u64,
        source: String,
    },
    /// Reading the model or building it.
    Loading,
    /// The model is ready; it came from `source` and took `ms` to build.
    Ready { source: String, ms: f64 },
    /// One stream update; `ms` of compute for `audio_s` seconds of new audio.
    Update {
        session: Session,
        text: String,
        continues: bool,
        pending: String,
        pending_continues: bool,
        language: Option<String>,
        ms: f64,
        audio_s: f64,
    },
    /// A finished stream's or a file's transcript (`audio_s` 0 for a stream's last pass: its
    /// audio was counted by the updates).
    Final {
        session: Session,
        text: String,
        language: Option<String>,
        words: Vec<TimedWord>,
        ms: f64,
        audio_s: f64,
    },
    /// Something failed; the text says what. `session` is `None` for the model's loading.
    Error {
        session: Option<Session>,
        message: String,
    },
}

impl FromEngine {
    fn error(session: Session, message: impl Into<String>) -> FromEngine {
        FromEngine::Error {
            session: Some(session),
            message: message.into(),
        }
    }

    fn engine_error(message: impl Into<String>) -> FromEngine {
        FromEngine::Error {
            session: None,
            message: message.into(),
        }
    }
}

/// Where the engine thread sends its replies (it also wakes the UI thread).
pub type Reply = Box<dyn Fn(FromEngine) + Send>;

/// The library's options for the UI's choices: the live preset with the language, beams,
/// keywords and timestamps the user picked (as the browser worker builds them).
pub fn options(ui: &UiOptions) -> Options {
    Options {
        language: ui.language_code().map(str::to_string),
        beams: ui.beam_count().max(1),
        keywords: ui.keyword_list(),
        word_timestamps: ui.word_timestamps,
        ..Options::streaming()
    }
}

fn timed(words: &[Word]) -> Vec<TimedWord> {
    words
        .iter()
        .map(|w| TimedWord {
            word: w.word.clone(),
            start: w.start,
            end: w.end,
        })
        .collect()
}

fn ms_since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

/// A live stream and what its final transcript needs.
struct Live<'w> {
    session: Session,
    stream: Stream<'w>,
    /// The audio's rate and its conversion to the model's (equal rates pass through).
    to_model: Option<(u32, StreamResampler)>,
    /// Every word the stream committed, for the final transcript's timings.
    words: Vec<Word>,
    /// The committed text, joined as the UI joins updates.
    text: String,
    word_timestamps: bool,
}

impl Live<'_> {
    /// `samples` at `rate` Hz as model-rate samples ready so far. A push at a new rate ends the
    /// old rate's audio first (its tail leads the output).
    fn convert(&mut self, samples: &[f32], rate: u32) -> Result<Vec<f32>, String> {
        if let Some((current, resampler)) = self.to_model.as_mut() {
            if *current == rate {
                return Ok(resampler.push(samples));
            }
        }
        let mut out = self.flush();
        let mut resampler = StreamResampler::new(rate, MODEL_RATE).map_err(|e| e.to_string())?;
        out.extend(resampler.push(samples));
        self.to_model = Some((rate, resampler));
        Ok(out)
    }

    /// The model-rate samples the conversion still owes.
    fn flush(&mut self) -> Vec<f32> {
        self.to_model
            .take()
            .map(|(_, mut r)| r.finish())
            .unwrap_or_default()
    }

    fn record(&mut self, update: &StreamUpdate) {
        record(&mut self.words, &mut self.text, update);
    }
}

/// Keep an update's words and text for the final transcript (joined as the UI joins updates).
fn record(words: &mut Vec<Word>, text: &mut String, update: &StreamUpdate) {
    words.extend(update.committed.iter().cloned());
    if !update.text.is_empty() {
        if !text.is_empty() && !update.continues {
            text.push(' ');
        }
        text.push_str(&update.text);
    }
}

/// The model and the live stream.
pub struct Engine<'w> {
    whistle: &'w Whistle,
    live: Option<Live<'w>>,
}

impl<'w> Engine<'w> {
    pub fn new(whistle: &'w Whistle) -> Engine<'w> {
        Engine {
            whistle,
            live: None,
        }
    }

    /// The open stream, if it is `session`'s.
    fn live(&mut self, session: Session) -> Option<&mut Live<'w>> {
        self.live.as_mut().filter(|l| l.session == session)
    }

    /// Answer one message; `None` when it has no reply (a stream opened or cancelled, audio for
    /// a stream that is no longer open).
    pub fn handle(&mut self, msg: ToEngine) -> Option<FromEngine> {
        match msg {
            ToEngine::Start {
                session,
                options: ui,
            } => {
                self.live = None;
                match self.whistle.stream(&options(&ui)) {
                    Ok(stream) => {
                        self.live = Some(Live {
                            session,
                            stream,
                            to_model: None,
                            words: Vec::new(),
                            text: String::new(),
                            word_timestamps: ui.word_timestamps,
                        });
                        None
                    }
                    Err(e) => Some(FromEngine::error(session, e.to_string())),
                }
            }
            ToEngine::Push {
                session,
                samples,
                rate,
            } => self.push(session, &samples, rate),
            ToEngine::Finish { session } => Some(self.finish(session)),
            ToEngine::Cancel { session } => {
                if self.live(session).is_some() {
                    self.live = None;
                }
                None
            }
            ToEngine::File {
                session,
                name,
                bytes,
                options: ui,
            } => Some(self.file(session, &name, &bytes, &ui)),
        }
    }

    fn push(&mut self, session: Session, samples: &[f32], rate: u32) -> Option<FromEngine> {
        let live = self.live(session)?;
        let t0 = Instant::now();
        let pushed = live
            .convert(samples, rate)
            .and_then(|audio| live.stream.push(&audio).map_err(|e| e.to_string()));
        let ms = ms_since(t0);
        match pushed {
            Ok(update) => {
                live.record(&update);
                Some(FromEngine::Update {
                    session,
                    text: update.text,
                    continues: update.continues,
                    pending: update.pending,
                    pending_continues: update.pending_continues,
                    language: update.language,
                    ms,
                    audio_s: samples.len() as f64 / rate.max(1) as f64,
                })
            }
            Err(e) => {
                // The stream ends here: the audio still queued for it is dropped, not answered
                // with one error each.
                self.live = None;
                Some(FromEngine::error(session, e))
            }
        }
    }

    fn finish(&mut self, session: Session) -> FromEngine {
        let Some(mut live) = self.live.take_if(|l| l.session == session) else {
            return FromEngine::error(session, "no stream is open");
        };
        let t0 = Instant::now();
        // The conversion's last few milliseconds, then the stream's last pass.
        let tail = live.flush();
        if !tail.is_empty() {
            match live.stream.push(&tail) {
                Ok(update) => live.record(&update),
                Err(e) => return FromEngine::error(session, e.to_string()),
            }
        }
        let Live {
            stream,
            mut words,
            mut text,
            word_timestamps,
            ..
        } = live;
        match stream.finish() {
            Ok(update) => {
                record(&mut words, &mut text, &update);
                FromEngine::Final {
                    session,
                    text,
                    language: update.language,
                    words: if word_timestamps {
                        timed(&words)
                    } else {
                        Vec::new()
                    },
                    ms: ms_since(t0),
                    audio_s: 0.0,
                }
            }
            Err(e) => FromEngine::error(session, e.to_string()),
        }
    }

    fn file(&self, session: Session, name: &str, wav: &[u8], ui: &UiOptions) -> FromEngine {
        let audio_s = whistle::audio::read_wav(wav)
            .map(|a| a.samples.len() as f64 / a.sample_rate.max(1) as f64)
            .unwrap_or(0.0);
        let t0 = Instant::now();
        match self.whistle.transcribe_wav(wav, &options(ui)) {
            Ok(t) => FromEngine::Final {
                session,
                text: t.text,
                language: t.language,
                words: timed(&t.words),
                ms: ms_since(t0),
                audio_s,
            },
            Err(e) => FromEngine::error(session, format!("{name}: {e}")),
        }
    }
}

/// Reports the engine gone if its thread panics (dropped while unwinding), so the UI shows an
/// error instead of waiting for a reply that will never come.
struct PanicGuard<'a>(&'a Reply);

impl Drop for PanicGuard<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            (self.0)(FromEngine::engine_error(
                "the transcription engine stopped unexpectedly",
            ));
        }
    }
}

/// Start the engine thread: it loads the model from `source`, then answers the returned
/// sender's messages through `reply`. An error says why the thread could not start.
pub fn spawn(source: ModelSource, reply: Reply) -> Result<Sender<ToEngine>, String> {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("whistle-demo-engine".into())
        .spawn(move || serve(&source, rx, &reply))
        .map_err(|e| format!("could not start the transcription engine: {e}"))?;
    Ok(tx)
}

/// The engine thread's body.
fn serve(source: &ModelSource, inbox: Receiver<ToEngine>, reply: &Reply) {
    let _guard = PanicGuard(reply);
    reply(FromEngine::Loading);
    let progress = |loaded, total, from: &str| {
        reply(FromEngine::Progress {
            loaded,
            total,
            source: from.to_string(),
        })
    };
    let (bytes, from) = match source.load(&progress) {
        Ok(found) => found,
        Err(e) => {
            reply(FromEngine::engine_error(format!(
                "could not load the model: {e}"
            )));
            return;
        }
    };
    reply(FromEngine::Loading);
    let t0 = Instant::now();
    let whistle = match Whistle::from_cact(&bytes) {
        Ok(w) => w,
        Err(e) => {
            reply(FromEngine::engine_error(format!(
                "could not build the model: {e}"
            )));
            return;
        }
    };
    drop(bytes);
    reply(FromEngine::Ready {
        source: from,
        ms: ms_since(t0),
    });
    let mut engine = Engine::new(&whistle);
    for msg in inbox {
        if let Some(answer) = engine.handle(msg) {
            reply(answer);
        }
    }
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
