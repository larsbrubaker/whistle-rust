//! Tests for the engine (`engine.rs`): a clip played through the capture seam (`WavCapture`,
//! as fast as it goes) becomes stream updates, then the final text; a clip at a device's rate
//! reads as it does at 16 kHz; audio for a session that has ended is dropped, and a failed push
//! ends its stream; a file is transcribed with its audio length and, when asked, its word
//! times; the UI's options map onto the library's.
//! They need the checkout's `models/whistle.cact` (`scripts/fetch-weights.sh`).

use std::sync::mpsc;
use std::sync::OnceLock;

use whistle::Whistle;
use whistle_demo::UiOptions;

use super::*;
use crate::capture::{Capture, CaptureEvent, WavCapture};
use crate::model::checkout_model;
use crate::test_support::{clip, SAMANTHA};

/// One model for every test here (building it takes a while).
fn whistle() -> &'static Whistle {
    static MODEL: OnceLock<Whistle> = OnceLock::new();
    MODEL.get_or_init(|| {
        let bytes = std::fs::read(checkout_model())
            .expect("models/whistle.cact (run scripts/fetch-weights.sh)");
        Whistle::from_cact(&bytes).expect("the model builds")
    })
}

/// Every chunk `capture` delivers, in order, until it ends.
fn captured(mut capture: WavCapture) -> Vec<(Vec<f32>, u32)> {
    let (tx, rx) = mpsc::channel();
    capture
        .start(Box::new(move |event| {
            let _ = tx.send(event);
        }))
        .expect("the capture starts");
    let mut chunks = Vec::new();
    for event in rx {
        match event {
            CaptureEvent::Audio { samples, rate } => chunks.push((samples, rate)),
            CaptureEvent::Ended(failure) => {
                assert_eq!(failure, None);
                break;
            }
        }
    }
    capture.stop();
    chunks
}

#[test]
fn a_streamed_clip_becomes_updates_then_the_final_text() {
    let mut engine = Engine::new(whistle());
    assert_eq!(
        engine.handle(ToEngine::Start {
            session: 1,
            options: UiOptions::default()
        }),
        None
    );
    let chunks = captured(WavCapture::from_wav(&clip("en_samantha"), false).expect("a WAV"));
    assert!(chunks.len() > 10, "100 ms chunks of a 4 s clip");
    let mut updates = 0;
    let mut shown = String::new();
    let mut audio = 0.0;
    for (samples, rate) in chunks {
        match engine.handle(ToEngine::Push {
            session: 1,
            samples,
            rate,
        }) {
            Some(FromEngine::Update {
                text,
                pending,
                audio_s,
                ms,
                ..
            }) => {
                updates += 1;
                audio += audio_s;
                assert!(ms >= 0.0);
                if !text.is_empty() || !pending.is_empty() {
                    shown = format!("{text}|{pending}");
                }
            }
            other => panic!("a push answers with an update, not {other:?}"),
        }
    }
    assert!(updates > 10);
    assert!(!shown.is_empty(), "the stream showed text before it ended");
    assert!(
        (audio - 4.0).abs() < 1.5,
        "the updates cover the clip ({audio} s)"
    );
    match engine.handle(ToEngine::Finish { session: 1 }) {
        Some(FromEngine::Final {
            text,
            language,
            words,
            audio_s,
            ..
        }) => {
            assert_eq!(text, SAMANTHA);
            assert_eq!(language.as_deref(), Some("en"));
            assert!(words.is_empty(), "no word timestamps asked for");
            assert_eq!(audio_s, 0.0);
        }
        other => panic!("finish answers with the final transcript, not {other:?}"),
    }
}

#[test]
fn a_stream_with_word_timestamps_finishes_with_the_words_times() {
    let mut engine = Engine::new(whistle());
    let ui = UiOptions {
        word_timestamps: true,
        ..UiOptions::default()
    };
    engine.handle(ToEngine::Start {
        session: 1,
        options: ui,
    });
    for (samples, rate) in captured(WavCapture::from_wav(&clip("en_samantha"), false).unwrap()) {
        engine.handle(ToEngine::Push {
            session: 1,
            samples,
            rate,
        });
    }
    let Some(FromEngine::Final { words, .. }) = engine.handle(ToEngine::Finish { session: 1 })
    else {
        panic!("a final transcript");
    };
    let joined: Vec<&str> = words.iter().map(|w| w.word.as_str()).collect();
    assert_eq!(joined.join(" "), SAMANTHA);
    assert!(words.windows(2).all(|w| w[0].start <= w[1].start));
}

#[test]
fn a_file_is_transcribed_with_its_length_and_word_times() {
    let mut engine = Engine::new(whistle());
    let ui = UiOptions {
        word_timestamps: true,
        ..UiOptions::default()
    };
    let msg = ToEngine::File {
        session: 4,
        name: "en_samantha.wav".into(),
        bytes: clip("en_samantha").into(),
        options: ui,
    };
    match engine.handle(msg) {
        Some(FromEngine::Final {
            session,
            text,
            words,
            audio_s,
            ms,
            ..
        }) => {
            assert_eq!(session, 4, "the reply names its session");
            assert_eq!(text, SAMANTHA);
            let joined: Vec<&str> = words.iter().map(|w| w.word.as_str()).collect();
            assert_eq!(joined.join(" "), SAMANTHA);
            assert!(audio_s > 3.0 && audio_s < 6.0, "{audio_s} s");
            assert!(ms > 0.0);
        }
        other => panic!("a final transcript, not {other:?}"),
    }
}

#[test]
fn a_file_that_is_not_a_wav_is_an_error_naming_it() {
    let mut engine = Engine::new(whistle());
    let reply = engine.handle(ToEngine::File {
        session: 2,
        name: "notes.txt".into(),
        bytes: b"hello".to_vec().into(),
        options: UiOptions::default(),
    });
    match reply {
        Some(FromEngine::Error {
            session: Some(2),
            message,
        }) => assert!(message.starts_with("notes.txt: "), "{message}"),
        other => panic!("an error, not {other:?}"),
    }
}

fn push(session: Session, rate: u32) -> ToEngine {
    ToEngine::Push {
        session,
        samples: vec![0.0; rate as usize / 10],
        rate,
    }
}

fn start(session: Session) -> ToEngine {
    ToEngine::Start {
        session,
        options: UiOptions::default(),
    }
}

#[test]
fn audio_for_a_session_that_ended_is_dropped() {
    let mut engine = Engine::new(whistle());
    assert_eq!(engine.handle(push(1, 16_000)), None, "no stream yet");
    engine.handle(start(1));
    assert!(matches!(
        engine.handle(push(1, 16_000)),
        Some(FromEngine::Update { session: 1, .. })
    ));
    // A new recording replaces the stream: the old one's queued audio goes nowhere.
    engine.handle(start(2));
    assert_eq!(engine.handle(push(1, 16_000)), None);
    assert!(matches!(
        engine.handle(push(2, 16_000)),
        Some(FromEngine::Update { session: 2, .. })
    ));
    assert_eq!(engine.handle(ToEngine::Cancel { session: 1 }), None);
    assert!(
        engine.handle(push(2, 16_000)).is_some(),
        "an old session's cancel leaves the new stream open"
    );
    engine.handle(ToEngine::Cancel { session: 2 });
    assert_eq!(engine.handle(push(2, 16_000)), None, "cancelled");
    assert_eq!(
        engine.handle(ToEngine::Finish { session: 2 }),
        Some(FromEngine::Error {
            session: Some(2),
            message: "no stream is open".into()
        })
    );
}

#[test]
fn a_failed_push_ends_its_stream_so_queued_audio_is_not_an_error_each() {
    let mut engine = Engine::new(whistle());
    engine.handle(start(3));
    // 500 Hz is outside the rates the resampler converts.
    match engine.handle(push(3, 500)) {
        Some(FromEngine::Error {
            session: Some(3),
            message,
        }) => assert!(message.contains("500"), "{message}"),
        other => panic!("an error, not {other:?}"),
    }
    assert_eq!(engine.handle(push(3, 500)), None);
    assert_eq!(engine.handle(push(3, 16_000)), None);
}

#[test]
fn the_ui_options_map_onto_the_live_preset() {
    let ui = UiOptions {
        language: 2,
        beams: 1,
        keywords: "PETG, MatterCAD".into(),
        word_timestamps: true,
    };
    let o = options(&ui);
    assert_eq!(o.language.as_deref(), Some("de"));
    assert_eq!(o.beams, 3);
    assert_eq!(o.keywords, vec!["PETG", "MatterCAD"]);
    assert!(o.word_timestamps);
    let live = whistle::Options::streaming();
    assert_eq!(o.stream_step, live.stream_step);
    assert_eq!(options(&UiOptions::default()).beams, 1);
}

/// The final text of `samples` at `rate` Hz streamed through the engine in 100 ms pushes, as
/// a microphone at that rate delivers them.
fn streamed_text(samples: &[f32], rate: u32) -> String {
    let mut engine = Engine::new(whistle());
    engine.handle(ToEngine::Start {
        session: 1,
        options: UiOptions::default(),
    });
    for chunk in samples.chunks(rate as usize / 10) {
        if let Some(FromEngine::Error { message, .. }) = engine.handle(ToEngine::Push {
            session: 1,
            samples: chunk.to_vec(),
            rate,
        }) {
            panic!("{message}");
        }
    }
    match engine.handle(ToEngine::Finish { session: 1 }) {
        Some(FromEngine::Final { text, .. }) => text,
        other => panic!("a final transcript, not {other:?}"),
    }
}

#[test]
fn a_clip_streamed_at_a_device_rate_reads_as_it_does_at_16_khz() {
    // Resampling each push on its own puts a seam at every chunk boundary; measured before
    // the fix, en_daniel at 44.1 or 48 kHz read "granting" for "printing".
    let audio = whistle::audio::read_wav(&clip("en_daniel")).expect("a WAV");
    assert_eq!(audio.sample_rate, 16_000);
    let reference = streamed_text(&audio.samples, 16_000);
    for rate in [44_100, 48_000] {
        let device = whistle::audio::resample(&audio.samples, 16_000, rate);
        assert_eq!(streamed_text(&device, rate), reference, "at {rate} Hz");
    }
}
