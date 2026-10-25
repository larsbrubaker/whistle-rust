//! Tests for the desktop host's message flow (`bridge.rs`) with a real engine thread
//! (`engine.rs`) on the checkout's model: a clip "recorded" through the capture seam
//! (`WavCapture` in place of a microphone) shows live text while listening and ends with the
//! final transcript once the capture runs out; a late reply from an old session never reaches
//! a new one; a file opened before the model is ready waits for it, and one opened while
//! something runs is ignored; a file that is not a WAV, a microphone that cannot start and an
//! engine that cannot start all end in an error the UI shows; without a capture Record is
//! unavailable.

use std::time::Duration;

use whistle_demo::{DemoHost, DemoState, Phase};

use super::*;
use crate::capture::WavCapture;
use crate::model::ModelSource;
use crate::test_support::{clip, local_model, Scratch, SAMANTHA};

/// Long enough for a debug build to load the model and transcribe a clip.
const PATIENCE: Duration = Duration::from_secs(120);

fn bridge(capture: Option<Box<dyn Capture>>) -> Rc<Bridge> {
    Bridge::start(
        DemoState::new(),
        local_model(),
        capture,
        "no microphone",
        Arc::new(|| {}),
    )
}

/// Drain until the model is ready.
fn until_ready(bridge: &Bridge) {
    let deadline = Instant::now() + PATIENCE;
    while !bridge.state().can_start() {
        assert!(Instant::now() < deadline, "the model never became ready");
        assert!(
            !matches!(bridge.state().phase(), Phase::Error(_)),
            "{:?}",
            bridge.state().phase()
        );
        bridge.wait(Duration::from_millis(200));
    }
}

#[test]
fn a_recorded_clip_shows_live_text_then_the_final_transcript() {
    let capture = WavCapture::from_wav(&clip("en_samantha"), false).expect("a WAV");
    let bridge = bridge(Some(Box::new(capture)));
    assert_eq!(bridge.state().mic_unavailable(), None);
    until_ready(&bridge);

    bridge.start_recording(&bridge.state().options());
    assert!(bridge.state().is_listening());
    let deadline = Instant::now() + PATIENCE;
    // This capture delivers the clip at once, so its end (and the Stop it causes) can reach
    // the UI before the engine's first update: the updates still in flight then show while
    // the stream finishes. Either way, live text shows before the final transcript.
    let mut live = Vec::new();
    loop {
        assert!(Instant::now() < deadline, "no final transcript");
        let Ok(event) = bridge.events.recv_timeout(Duration::from_millis(200)) else {
            continue;
        };
        bridge.apply(event);
        if bridge.outcomes() > 0 {
            break;
        }
        let s = bridge.state();
        if !(s.committed().is_empty() && s.pending().is_empty()) {
            live.push(format!("{}|{}", s.committed(), s.pending()));
        }
    }
    assert!(!live.is_empty(), "text showed before the final transcript");
    let s = bridge.state();
    assert_eq!(
        s.phase(),
        Phase::Ready,
        "the capture ended, so the stream finished"
    );
    assert_eq!(s.committed(), SAMANTHA);
    assert_eq!(s.pending(), "");
    assert!(s
        .status_line()
        .starts_with("Ready \u{b7} en \u{b7} last update"));
}

#[test]
fn stopping_a_recording_finishes_the_stream_with_what_was_heard() {
    // Real time, so Stop lands mid-clip, as a user's would.
    let capture = WavCapture::from_wav(&clip("en_samantha"), true).expect("a WAV");
    let bridge = bridge(Some(Box::new(capture)));
    until_ready(&bridge);
    bridge.start_recording(&bridge.state().options());
    std::thread::sleep(Duration::from_millis(1500));
    bridge.drain();
    bridge.stop_recording();
    assert_eq!(bridge.state().phase(), Phase::Transcribing);
    assert!(bridge.wait_for_outcomes(1, PATIENCE), "no final transcript");
    let text = bridge.state().committed();
    assert_eq!(bridge.state().phase(), Phase::Ready);
    // What a pass makes of the word cut off at Stop is the model's business; the host's is
    // that the stream finished with the audio heard so far, and only that.
    let words = text.split_whitespace().count();
    assert!(
        text.starts_with("Make the") && words < SAMANTHA.split_whitespace().count(),
        "the first part of the clip only: {text:?}"
    );
}

#[test]
fn a_file_opened_before_the_model_is_ready_waits_for_it() {
    let dir = Scratch::new("open-early");
    let wav = dir.path().join("en_samantha.wav");
    std::fs::write(&wav, clip("en_samantha")).expect("write");
    let bridge = bridge(None);
    bridge.at_startup(Startup::Open(wav));
    assert!(bridge.wait_for_outcomes(1, PATIENCE), "no final transcript");
    let s = bridge.state();
    assert_eq!(s.committed(), SAMANTHA);
    assert!(s.status_line().contains("RTF"), "{}", s.status_line());
}

#[test]
fn a_dropped_file_that_is_not_a_wav_ends_in_an_error() {
    let bridge = bridge(None);
    until_ready(&bridge);
    bridge.transcribe_file("notes.txt", b"hello".to_vec().into(), &UiOptions::default());
    assert_eq!(bridge.state().phase(), Phase::Transcribing);
    assert!(bridge.wait_for_outcomes(1, PATIENCE));
    match bridge.state().phase() {
        Phase::Error(e) => assert!(e.starts_with("notes.txt: "), "{e}"),
        other => panic!("an error, not {other:?}"),
    }
    assert!(bridge.state().can_start(), "the model is still usable");
}

#[test]
fn without_a_capture_record_is_unavailable_and_does_nothing() {
    let bridge = bridge(None);
    assert_eq!(
        bridge.state().mic_unavailable().as_deref(),
        Some("no microphone")
    );
    until_ready(&bridge);
    assert!(!bridge.state().can_record());
    bridge.start_recording(&UiOptions::default());
    assert_eq!(bridge.state().phase(), Phase::Ready);
}

#[test]
fn a_model_that_cannot_load_is_an_error_the_ui_shows() {
    let missing = std::env::temp_dir().join("whistle-demo-native-no-such-model.cact");
    let bridge = Bridge::start(
        DemoState::new(),
        ModelSource {
            path: Some(missing.clone()),
            ..local_model()
        },
        None,
        "no microphone",
        Arc::new(|| {}),
    );
    assert!(bridge.wait_for_outcomes(1, PATIENCE));
    match bridge.state().phase() {
        Phase::Error(e) => {
            assert!(e.starts_with("could not load the model: "), "{e}");
            assert!(e.contains(&missing.display().to_string()), "{e}");
        }
        other => panic!("an error, not {other:?}"),
    }
    assert!(!bridge.state().can_start());
}

/// A capture that starts with one chunk of silence and delivers nothing more (a microphone in a
/// quiet room, as far as the transcript goes), or fails to start.
struct QuietCapture {
    fails: bool,
}

impl Capture for QuietCapture {
    fn start(&mut self, mut sink: crate::capture::Sink) -> Result<(), String> {
        if self.fails {
            Err("no input device".into())
        } else {
            sink(CaptureEvent::Audio {
                samples: vec![0.0; 1600],
                rate: 16_000,
            });
            Ok(())
        }
    }
    fn stop(&mut self) {}
}

fn update(session: Session, text: &str) -> Event {
    Event::Engine(FromEngine::Update {
        session,
        text: text.into(),
        continues: false,
        pending: String::new(),
        pending_continues: false,
        language: None,
        ms: 1.0,
        audio_s: 0.1,
    })
}

#[test]
fn a_late_update_from_an_old_stream_does_not_reach_a_new_recording() {
    let bridge = bridge(Some(Box::new(QuietCapture { fails: false })));
    until_ready(&bridge);
    bridge.start_recording(&UiOptions::default());
    // Its audio has come: Listening (so Stop finishes the stream).
    bridge.drain();
    assert_eq!(bridge.state().phase(), Phase::Listening);
    let old = bridge.session.get();
    bridge.stop_recording();
    assert!(
        bridge.wait_for_outcomes(1, PATIENCE),
        "the first stream finished"
    );
    bridge.start_recording(&UiOptions::default());
    bridge.drain();
    assert_eq!(bridge.state().phase(), Phase::Listening);

    bridge.events_tx.send(update(old, "stale words")).unwrap();
    bridge.drain();
    assert_eq!(
        bridge.state().committed(),
        "",
        "the old stream's update was dropped"
    );
    bridge
        .events_tx
        .send(update(bridge.session.get(), "fresh words"))
        .unwrap();
    bridge.drain();
    assert_eq!(bridge.state().committed(), "fresh words");
}

#[test]
fn a_file_opened_while_another_is_transcribed_is_ignored_like_a_drop() {
    let dir = Scratch::new("open-busy");
    let wav = dir.path().join("en_daniel.wav");
    std::fs::write(&wav, clip("en_daniel")).expect("write");
    let bridge = bridge(None);
    until_ready(&bridge);
    bridge.transcribe_file(
        "en_samantha.wav",
        clip("en_samantha").into(),
        &UiOptions::default(),
    );
    bridge.at_startup(Startup::Open(wav));
    assert!(bridge.startup.borrow().is_empty(), "not queued");
    assert!(bridge.wait_for_outcomes(1, PATIENCE));
    assert_eq!(bridge.state().committed(), SAMANTHA);
    assert!(
        !bridge.wait_for_outcomes(2, Duration::from_secs(3)),
        "no second transcription"
    );
}

#[test]
fn a_microphone_that_cannot_start_is_an_error_the_ui_shows() {
    let bridge = bridge(Some(Box::new(QuietCapture { fails: true })));
    until_ready(&bridge);
    bridge.start_recording(&UiOptions::default());
    assert_eq!(
        bridge.state().phase(),
        Phase::Error("could not start the microphone: no input device".into())
    );
    assert_eq!(bridge.outcomes(), 1, "counted, so a scripted run sees it");
    assert!(bridge.state().can_start(), "files still work");
}

#[test]
fn an_engine_that_cannot_start_is_an_error_the_ui_shows() {
    let bridge = Bridge::assemble(
        DemoState::new(),
        Err("could not start the transcription engine: out of threads".into()),
        {
            let (tx, rx) = mpsc::channel();
            (rx, tx)
        },
        None,
        "no microphone",
        Arc::new(|| {}),
    );
    assert_eq!(
        bridge.state().phase(),
        Phase::Error("could not start the transcription engine: out of threads".into())
    );
    assert!(!bridge.state().can_start());
}
