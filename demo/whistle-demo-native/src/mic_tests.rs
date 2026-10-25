//! Tests for the system microphone as a capture (`mic.rs`), over agg-mic's `Microphone` with a
//! fake backend in place of the device, so no microphone is opened and no permission is asked:
//! with a real engine thread on the checkout's model (`bridge.rs`, `engine.rs`), the
//! microphone's chunks reach the engine and show live text, a stop delivers the device's last
//! audio before the stream finishes, a microphone that fails mid-recording ends the recording
//! with agg-mic's message, and a refused permission shows agg-mic's denied text. Record says
//! Listening only once audio flows (until then it waits for the microphone, through the
//! permission prompt), and a Stop before any audio returns to Ready without a transcript.
//! Without the engine: a stop while the permission prompt is up returns at once and nothing
//! reaches the sink after it, a Record while the last start still waits on its prompt says so,
//! and a platform without a microphone leaves Record unavailable with agg-mic's note.

use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use agg_mic::{
    Backend, Delivery, MicError, Microphone, DISCONNECTED_MESSAGE, MAC_PERMISSION_DENIED_MESSAGE,
    UNSUPPORTED_MESSAGE,
};
use whistle_demo::{DemoHost, DemoState, Phase};

use super::*;
use crate::bridge::Bridge;
use crate::capture::{Capture, CaptureEvent};
use crate::test_support::{clip, local_model, SAMANTHA};

/// Long enough for a debug build to load the model and transcribe a clip.
const PATIENCE: Duration = Duration::from_secs(120);

/// The fake device's knobs, and what the test hands it, shared with its backend.
#[derive(Default)]
struct Device {
    unavailable: bool,
    /// The start's refusal (a permission denied), if any.
    refuse: Option<MicError>,
    /// A start waits for a message here, as a start waits on a permission prompt.
    prompt: Mutex<Option<Receiver<()>>>,
    /// The running recording's delivery, as a capture thread holds it; `started` signals it.
    delivery: Mutex<Option<Delivery>>,
    started: Condvar,
    /// Audio the device still holds when stopped, delivered inside the stop (a platform
    /// capture's last I/O cycle), with its rate.
    tail: Mutex<Option<(Vec<f32>, u32)>>,
}

impl Device {
    /// The recording's delivery, once the microphone's start (on the capture's own thread)
    /// has reached the backend.
    fn delivery(&self) -> Delivery {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut slot = self.delivery.lock().unwrap();
        while slot.is_none() {
            let left = deadline.saturating_duration_since(Instant::now());
            assert!(!left.is_zero(), "the microphone never started");
            slot = self.started.wait_timeout(slot, left).unwrap().0;
        }
        slot.clone().unwrap()
    }

    /// `samples` at `rate` Hz in 100 ms chunks, as the device would hand them over.
    fn speak(&self, samples: &[f32], rate: u32) {
        let delivery = self.delivery();
        for chunk in samples.chunks((rate / 10) as usize) {
            delivery.deliver(chunk, rate);
        }
    }
}

struct FakeBackend(Arc<Device>);

impl Backend for FakeBackend {
    fn is_available(&self) -> bool {
        !self.0.unavailable
    }

    fn start(&self, delivery: Delivery) -> Result<(), MicError> {
        let prompt = self.0.prompt.lock().unwrap().take();
        if let Some(prompt) = prompt {
            let _ = prompt.recv();
        }
        if let Some(refusal) = &self.0.refuse {
            return Err(refusal.clone());
        }
        *self.0.delivery.lock().unwrap() = Some(delivery);
        self.0.started.notify_all();
        Ok(())
    }

    fn stop(&self) {
        if let Some(delivery) = self.0.delivery.lock().unwrap().take() {
            if let Some((samples, rate)) = self.0.tail.lock().unwrap().take() {
                delivery.deliver(&samples, rate);
            }
        }
    }
}

/// A microphone over a fake device with `device`'s knobs, and the device.
fn microphone(device: Device) -> (Microphone, Arc<Device>) {
    let device = Arc::new(device);
    (
        Microphone::with_backend(FakeBackend(Arc::clone(&device))),
        device,
    )
}

/// A bridge whose Record button records from `microphone`, on the checkout's model.
fn bridge(microphone: Microphone) -> Rc<Bridge> {
    let capture = MicCapture::new(microphone).expect("the fake microphone is available");
    Bridge::start(
        DemoState::new(),
        local_model(),
        Some(Box::new(capture)),
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

/// en_samantha's samples and rate.
fn samantha() -> (Vec<f32>, u32) {
    let audio = whistle::audio::read_wav(&clip("en_samantha")).expect("a WAV");
    (audio.samples, audio.sample_rate)
}

#[test]
fn microphone_chunks_reach_the_engine_and_show_live_text() {
    let (microphone, device) = microphone(Device::default());
    let bridge = bridge(microphone);
    assert_eq!(bridge.state().mic_unavailable(), None);
    until_ready(&bridge);

    bridge.start_recording(&bridge.state().options());
    let (samples, rate) = samantha();
    device.speak(&samples, rate);
    // The device never ends the recording, so text shows while it still listens.
    let deadline = Instant::now() + PATIENCE;
    loop {
        assert!(Instant::now() < deadline, "no live text");
        bridge.wait(Duration::from_millis(200));
        let s = bridge.state();
        assert_eq!(s.phase(), Phase::Listening);
        if !(s.committed().is_empty() && s.pending().is_empty()) {
            break;
        }
    }

    bridge.stop_recording();
    assert!(bridge.wait_for_outcomes(1, PATIENCE), "no final transcript");
    assert_eq!(bridge.state().phase(), Phase::Ready);
    assert_eq!(bridge.state().committed(), SAMANTHA);
}

#[test]
fn a_stop_delivers_the_microphones_last_audio_before_the_stream_finishes() {
    let (microphone, device) = microphone(Device::default());
    let bridge = bridge(microphone);
    until_ready(&bridge);

    bridge.start_recording(&bridge.state().options());
    // "Make the cube 20 mm wide" while recording; "and round the top edges." is still in the
    // device when Stop comes, and only its stop hands it over.
    let (samples, rate) = samantha();
    let (said, held) = samples.split_at(samples.len() / 2);
    *device.tail.lock().unwrap() = Some((held.to_vec(), rate));
    device.speak(said, rate);
    bridge.stop_recording();
    assert_eq!(bridge.state().phase(), Phase::Transcribing);
    assert!(
        device.tail.lock().unwrap().is_none(),
        "the stop took the tail"
    );
    assert!(bridge.wait_for_outcomes(1, PATIENCE), "no final transcript");
    assert_eq!(
        bridge.state().committed(),
        SAMANTHA,
        "the tail reached the stream before it finished"
    );
}

#[test]
fn a_microphone_that_fails_while_recording_ends_it_with_the_message() {
    let (microphone, device) = microphone(Device::default());
    let bridge = bridge(microphone);
    until_ready(&bridge);

    bridge.start_recording(&bridge.state().options());
    let (samples, rate) = samantha();
    device.speak(&samples[..samples.len() / 2], rate);
    device
        .delivery()
        .fail(MicError::NoDevice(DISCONNECTED_MESSAGE.to_owned()));
    assert!(bridge.wait_for_outcomes(1, PATIENCE), "the failure showed");
    assert_eq!(
        bridge.state().phase(),
        Phase::Error(DISCONNECTED_MESSAGE.into())
    );
    assert!(bridge.state().can_record(), "Record can be tried again");
    assert!(
        !bridge.wait_for_outcomes(2, Duration::from_secs(2)),
        "the cancelled stream has no final transcript"
    );
}

#[test]
fn a_refused_permission_shows_the_denied_text() {
    let (microphone, _device) = microphone(Device {
        refuse: Some(MicError::PermissionDenied(
            MAC_PERMISSION_DENIED_MESSAGE.to_owned(),
        )),
        ..Device::default()
    });
    let bridge = bridge(microphone);
    until_ready(&bridge);

    bridge.start_recording(&bridge.state().options());
    assert!(bridge.wait_for_outcomes(1, PATIENCE), "the refusal showed");
    assert_eq!(
        bridge.state().phase(),
        Phase::Error(MAC_PERMISSION_DENIED_MESSAGE.into())
    );
    assert!(
        bridge.state().can_record(),
        "Record can be tried again once access is turned on"
    );
}

#[test]
fn a_stop_while_the_permission_prompt_is_up_returns_at_once_and_nothing_follows() {
    let (answer, prompt): (Sender<()>, Receiver<()>) = mpsc::channel();
    let (microphone, _device) = microphone(Device {
        prompt: Mutex::new(Some(prompt)),
        ..Device::default()
    });
    let mut capture = MicCapture::new(microphone).expect("available");
    let (tx, events) = mpsc::channel::<CaptureEvent>();
    capture
        .start(Box::new(move |event| {
            let _ = tx.send(event);
        }))
        .expect("the start thread runs");
    let began = Instant::now();
    capture.stop();
    assert!(
        began.elapsed() < Duration::from_secs(1),
        "the stop did not wait for the prompt"
    );
    // The prompt is answered after the stop: the start ends as stopped and tells no one.
    answer.send(()).unwrap();
    assert!(
        events.recv_timeout(Duration::from_secs(5)).is_err(),
        "nothing reached the sink after the stop"
    );
}

#[test]
fn without_microphone_support_record_is_unavailable_with_agg_mics_note() {
    let (microphone, _device) = microphone(Device {
        unavailable: true,
        ..Device::default()
    });
    let note = MicCapture::new(microphone).err().expect("unavailable");
    assert_eq!(note, UNSUPPORTED_MESSAGE);
}

/// Drain events for `span` (a phase that should not change has had its chance to).
fn drain_for(bridge: &Bridge, span: Duration) {
    let until = Instant::now() + span;
    while Instant::now() < until {
        bridge.wait(Duration::from_millis(50));
    }
}

/// A device whose start waits on a permission prompt, and the prompt's answer.
fn prompting() -> (Microphone, Arc<Device>, Sender<()>) {
    let (answer, prompt) = mpsc::channel();
    let (microphone, device) = microphone(Device {
        prompt: Mutex::new(Some(prompt)),
        ..Device::default()
    });
    (microphone, device, answer)
}

#[test]
fn record_waits_for_the_microphone_and_says_listening_only_once_audio_flows() {
    let (microphone, device, answer) = prompting();
    let bridge = bridge(microphone);
    until_ready(&bridge);

    bridge.start_recording(&bridge.state().options());
    let waiting = |why: &str| {
        let s = bridge.state();
        assert_eq!(s.phase(), Phase::RequestingMic, "{why}");
        assert_eq!(
            s.status_line(),
            "Asking for the microphone\u{2026}",
            "{why}"
        );
        assert!(s.is_listening(), "{why}: Record is a Stop button meanwhile");
    };
    waiting("the prompt is up");
    drain_for(&bridge, Duration::from_millis(300));
    waiting("the prompt is still up");
    answer.send(()).unwrap();
    device.delivery();
    drain_for(&bridge, Duration::from_millis(300));
    waiting("the device is open but no audio has come");

    let (samples, rate) = samantha();
    device.speak(&samples[..(rate / 10) as usize], rate);
    let deadline = Instant::now() + PATIENCE;
    while bridge.state().phase() == Phase::RequestingMic {
        assert!(Instant::now() < deadline, "never listening");
        bridge.wait(Duration::from_millis(200));
    }
    assert_eq!(bridge.state().phase(), Phase::Listening);
}

#[test]
fn a_stop_before_any_audio_returns_to_ready_without_a_transcript() {
    let (microphone, _device, answer) = prompting();
    let bridge = bridge(microphone);
    until_ready(&bridge);

    bridge.start_recording(&bridge.state().options());
    assert_eq!(bridge.state().phase(), Phase::RequestingMic);
    bridge.stop_recording();
    assert_eq!(bridge.state().phase(), Phase::Ready);
    // The prompt is answered after the stop: the recording is over and stays over.
    answer.send(()).unwrap();
    assert!(
        !bridge.wait_for_outcomes(1, Duration::from_secs(2)),
        "nothing was heard, so there is no transcript"
    );
    assert_eq!(bridge.state().phase(), Phase::Ready);
    assert!(bridge.state().can_record());
}

#[test]
fn record_while_the_last_start_still_waits_on_its_prompt_says_so() {
    let (microphone, _device, answer) = prompting();
    let mut capture = MicCapture::new(microphone).expect("available");
    capture.start(Box::new(|_| {})).expect("the first start");
    capture.stop();
    assert_eq!(
        capture.start(Box::new(|_| {})),
        Err(STILL_STARTING_MESSAGE.to_owned())
    );
    // Once the prompt is answered the last start ends, and Record works again.
    answer.send(()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match capture.start(Box::new(|_| {})) {
            Ok(()) => break,
            Err(e) => {
                assert_eq!(e, STILL_STARTING_MESSAGE);
                assert!(Instant::now() < deadline, "the last start never ended");
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
    capture.stop();
}
