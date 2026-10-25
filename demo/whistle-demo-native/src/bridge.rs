//! The UI thread's side of the desktop demo: [`Bridge`] is the whistle-demo UI's [`DemoHost`]
//! (record, stop, pick or drop a file, open a link) and turns what the engine thread
//! (`engine.rs`) and the capture (`capture.rs`) report into [`DemoState`] changes, as the
//! browser demo's page glue (`demo/web/src/main.ts`) does between its worker and the UI.
//!
//! Replies and capture events arrive on one channel; each sender also calls the [`Wake`] the
//! shell gave (`main.rs`: the UI thread's queue), so an idle window wakes and the shell host
//! (`shell_host.rs`) drains them. A file to open, or a recording to start, before the model has
//! loaded waits for it; once it has, a request while something runs is ignored, as a drop is.
//!
//! Each recording and file is a session with its own number (`engine.rs`): replies to any but
//! the current one are dropped, so a late update from an old stream never reaches a new one.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use whistle_demo::{DemoHost, DemoState, Phase, UiOptions};

use crate::capture::{Capture, CaptureEvent};
use crate::engine::{self, FromEngine, Reply, Session, ToEngine};
use crate::model::ModelSource;

/// Wakes the UI thread (safe from any thread).
pub type Wake = Arc<dyn Fn() + Send + Sync>;

/// What reaches the UI thread.
#[derive(Debug)]
pub enum Event {
    Engine(FromEngine),
    /// The capture ended on its own, with the failure if there was one.
    CaptureEnded(Option<String>),
}

/// What to do once the model is ready (from the command line or the OS).
#[derive(Debug, Clone, PartialEq)]
pub enum Startup {
    Open(PathBuf),
    Record,
}

/// See the module docs.
pub struct Bridge {
    state: DemoState,
    engine: Sender<ToEngine>,
    events: Receiver<Event>,
    events_tx: Sender<Event>,
    wake: Wake,
    capture: RefCell<Option<Box<dyn Capture>>>,
    capturing: Cell<bool>,
    /// Waiting for the model.
    startup: RefCell<Vec<Startup>>,
    /// Finished transcripts and errors so far (the screenshot run waits for one).
    outcomes: Cell<u64>,
    /// The current recording's or file's number; replies to others are stale.
    session: Cell<Session>,
}

impl Bridge {
    /// Start the engine thread on `source`; `capture` is the live audio source, or `None`
    /// with `mic_note` saying why Record is unavailable.
    pub fn start(
        state: DemoState,
        source: ModelSource,
        capture: Option<Box<dyn Capture>>,
        mic_note: &str,
        wake: Wake,
    ) -> Rc<Bridge> {
        let (events_tx, events) = mpsc::channel();
        let reply: Reply = {
            let (tx, wake) = (events_tx.clone(), Arc::clone(&wake));
            Box::new(move |msg| {
                if tx.send(Event::Engine(msg)).is_ok() {
                    wake();
                }
            })
        };
        let spawned = engine::spawn(source, reply);
        Bridge::assemble(state, spawned, (events, events_tx), capture, mic_note, wake)
    }

    /// The bridge over the engine `spawned` started, or showing why it could not start.
    fn assemble(
        state: DemoState,
        spawned: Result<Sender<ToEngine>, String>,
        (events, events_tx): (Receiver<Event>, Sender<Event>),
        capture: Option<Box<dyn Capture>>,
        mic_note: &str,
        wake: Wake,
    ) -> Rc<Bridge> {
        state.set_mic_unavailable(capture.is_none().then(|| mic_note.to_string()));
        let (engine, failure) = match spawned {
            Ok(engine) => (engine, None),
            // A sender to nowhere: every request then reports the engine stopped.
            Err(e) => (mpsc::channel().0, Some(e)),
        };
        let bridge = Rc::new(Bridge {
            state,
            engine,
            events,
            events_tx,
            wake,
            capture: RefCell::new(capture),
            capturing: Cell::new(false),
            startup: RefCell::new(Vec::new()),
            outcomes: Cell::new(0),
            session: Cell::new(0),
        });
        if let Some(e) = failure {
            bridge.fail(e);
        }
        bridge
    }

    /// A new session's number (it becomes the current one).
    fn next_session(&self) -> Session {
        self.session.set(self.session.get() + 1);
        self.session.get()
    }

    #[cfg(test)]
    pub fn state(&self) -> &DemoState {
        &self.state
    }

    /// Transcripts finished and errors shown so far.
    pub fn outcomes(&self) -> u64 {
        self.outcomes.get()
    }

    /// Do `what` once the model has loaded: now, when it has (ignored if something is running,
    /// as a drop is), else when it is ready.
    pub fn at_startup(&self, what: Startup) {
        if self.state.model_loaded() {
            self.run_startup(what);
        } else {
            self.startup.borrow_mut().push(what);
        }
    }

    fn run_startup(&self, what: Startup) {
        match what {
            Startup::Open(path) => self.open_path(&path),
            Startup::Record => {
                if self.state.can_record() {
                    self.start_recording(&self.state.options());
                }
            }
        }
    }

    /// Transcribe the WAV at `path` with the current options (the OS's open request, the
    /// command line, the file dialog).
    pub fn open_path(&self, path: &Path) {
        match std::fs::read(path) {
            Ok(bytes) => {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                self.transcribe_file(&name, bytes.into(), &self.state.options());
            }
            Err(e) => self.fail(format!("could not read {}: {e}", path.display())),
        }
    }

    /// Apply every waiting event; whether there were any.
    pub fn drain(&self) -> bool {
        let mut any = false;
        while let Ok(event) = self.events.try_recv() {
            self.apply(event);
            any = true;
        }
        any
    }

    /// Wait up to `timeout` for an event, then [`Bridge::drain`]; whether there were any.
    pub fn wait(&self, timeout: Duration) -> bool {
        match self.events.recv_timeout(timeout) {
            Ok(event) => {
                self.apply(event);
                self.drain();
                true
            }
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => false,
        }
    }

    /// Wait until `outcomes` reaches `count` or `timeout` passes; whether it did.
    pub fn wait_for_outcomes(&self, count: u64, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while self.outcomes() < count {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() || !self.wait(left) {
                return self.outcomes() >= count;
            }
        }
        true
    }

    fn apply(&self, event: Event) {
        match event {
            Event::Engine(msg) => self.apply_engine(msg),
            Event::CaptureEnded(None) => self.stop_recording(),
            Event::CaptureEnded(Some(e)) => {
                self.stop_capture();
                self.send(ToEngine::Cancel {
                    session: self.session.get(),
                });
                self.fail(e);
            }
        }
    }

    fn apply_engine(&self, msg: FromEngine) {
        let s = &self.state;
        let stale = match &msg {
            FromEngine::Update { session, .. } | FromEngine::Final { session, .. } => {
                *session != self.session.get()
            }
            FromEngine::Error {
                session: Some(session),
                ..
            } => *session != self.session.get(),
            _ => false,
        };
        if stale {
            return;
        }
        match msg {
            FromEngine::Progress {
                loaded,
                total,
                source,
            } => s.set_download(loaded, total, &source),
            FromEngine::Loading => s.set_phase(Phase::Loading),
            FromEngine::Ready { source, ms } => {
                eprintln!("whistle-demo-native: model from {source}, built in {ms:.0} ms");
                s.set_phase(Phase::Ready);
                let waiting = std::mem::take(&mut *self.startup.borrow_mut());
                for what in waiting {
                    self.run_startup(what);
                }
            }
            FromEngine::Update {
                session: _,
                text,
                continues,
                pending,
                pending_continues,
                language,
                ms,
                audio_s,
            } => s.apply_update(
                &text,
                continues,
                (&pending, pending_continues),
                language,
                ms,
                audio_s,
            ),
            FromEngine::Final {
                session: _,
                text,
                language,
                words,
                ms,
                audio_s,
            } => {
                s.set_final(&text, language, words, ms, audio_s);
                // The transcript on the terminal too, for scripted runs.
                println!("{text}\n{}", s.status_line());
                self.outcomes.set(self.outcomes.get() + 1);
            }
            FromEngine::Error { message, .. } => {
                self.stop_capture();
                self.fail(message);
            }
        }
    }

    fn fail(&self, message: String) {
        eprintln!("whistle-demo-native: {message}");
        self.state.set_phase(Phase::Error(message));
        self.outcomes.set(self.outcomes.get() + 1);
    }

    fn send(&self, msg: ToEngine) {
        if self.engine.send(msg).is_err() {
            self.fail("the transcription engine stopped".into());
        }
    }

    fn stop_capture(&self) {
        if self.capturing.replace(false) {
            if let Some(capture) = self.capture.borrow_mut().as_mut() {
                capture.stop();
            }
        }
    }
}

impl DemoHost for Bridge {
    fn start_recording(&self, options: &UiOptions) {
        let mut capture = self.capture.borrow_mut();
        let Some(capture) = capture.as_mut() else {
            return;
        };
        self.state.set_phase(Phase::RequestingMic);
        let session = self.next_session();
        // The stream opens before any audio can reach the engine (one inbox, in order).
        self.send(ToEngine::Start {
            session,
            options: options.clone(),
        });
        let (engine, events, wake) = (
            self.engine.clone(),
            self.events_tx.clone(),
            Arc::clone(&self.wake),
        );
        let sink = Box::new(move |event| match event {
            CaptureEvent::Audio { samples, rate } => {
                // A closed engine is reported by the bridge's own next send.
                let _ = engine.send(ToEngine::Push {
                    session,
                    samples,
                    rate,
                });
            }
            CaptureEvent::Ended(failure) => {
                if events.send(Event::CaptureEnded(failure)).is_ok() {
                    wake();
                }
            }
        });
        match capture.start(sink) {
            Ok(()) => {
                self.capturing.set(true);
                self.state.start_listening();
            }
            Err(e) => {
                self.send(ToEngine::Cancel { session });
                self.fail(format!("could not start the microphone: {e}"));
            }
        }
    }

    fn stop_recording(&self) {
        if !self.capturing.get() {
            if self.state.is_listening() {
                self.state.set_phase(Phase::Ready);
            }
            return;
        }
        // Every chunk is in the engine's inbox once the capture stopped, so Finish follows
        // the last of the audio.
        self.stop_capture();
        self.state.set_phase(Phase::Transcribing);
        self.send(ToEngine::Finish {
            session: self.session.get(),
        });
    }

    fn pick_file(&self, _options: &UiOptions) {
        // Blocking: the frame loop waits while the dialog is up. The picked file goes through
        // `open_path`, which reads the options again (as the browser's picker does).
        let picked = rfd::FileDialog::new()
            .set_title("Transcribe a WAV file")
            .add_filter("WAV audio", &["wav", "WAV"])
            .pick_file();
        if let Some(path) = picked {
            self.open_path(&path);
        }
    }

    fn transcribe_file(&self, name: &str, bytes: Arc<[u8]>, options: &UiOptions) {
        if !self.state.can_start() {
            return;
        }
        self.state.start_file(name);
        self.send(ToEngine::File {
            session: self.next_session(),
            name: name.to_string(),
            bytes,
            options: options.clone(),
        });
    }

    fn open_url(&self, url: &str) {
        if let Err(e) = open_in_browser(url) {
            eprintln!("whistle-demo-native: could not open {url}: {e}");
        }
    }
}

/// Open `url` in the default browser.
fn open_in_browser(url: &str) -> std::io::Result<()> {
    let mut command = if cfg!(target_os = "macos") {
        std::process::Command::new("open")
    } else if cfg!(windows) {
        // `start`'s first quoted argument is the window title, hence the empty one.
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "start", ""]);
        c
    } else {
        std::process::Command::new("xdg-open")
    };
    command.arg(url).spawn().map(|_| ())
}

#[cfg(test)]
#[path = "bridge_tests.rs"]
mod tests;
