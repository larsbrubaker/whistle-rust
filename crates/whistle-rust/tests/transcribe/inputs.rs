// Caller inputs the decode must never see: a non-finite sample is an input error naming it,
// before the detector runs, in every VAD mode; audio shorter than one 10 ms feature hop (no
// valid frame for the decoder to attend to) is an empty transcript in every mode, while one hop
// or more still decodes; a WAV file that cannot be read is `Error::Audio`; audio over the
// limits is rejected before it is resampled (30 s with the gate off or gate only, 10 minutes
// before trimming).

use std::error::Error as _;

use whistle::audio::{read_wav, AudioError};
use whistle::{Error, Options, VadMode, MAX_INPUT_SAMPLES};

use crate::shared::{wav, whistle};

const MODES: [VadMode; 3] = [VadMode::Off, VadMode::Gate, VadMode::GateAndTrim];

fn options(vad: VadMode) -> Options {
    Options {
        vad,
        ..Options::default()
    }
}

/// A 16-bit mono PCM WAV file of `samples` at `rate`.
fn wav_bytes(samples: &[f32], rate: u32) -> Vec<u8> {
    let data = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&((s * 32767.0) as i16).to_le_bytes());
    }
    out
}

#[test]
fn a_non_finite_sample_is_an_input_error_naming_it_in_every_mode() {
    let clean = read_wav(&wav("en_samantha")).unwrap().samples;
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for vad in MODES {
            let mut samples = clean.clone();
            samples[12_345] = bad;
            let err = whistle().transcribe(&samples, &options(vad)).unwrap_err();
            assert!(
                matches!(&err, Error::Input(m) if m.contains("12345") && m.contains("finite")),
                "{bad} {vad:?}: {err}"
            );
        }
    }
}

#[test]
fn audio_shorter_than_one_hop_is_an_empty_transcript_in_every_mode() {
    let speech = read_wav(&wav("en_samantha")).unwrap().samples;
    for len in [1, 100, 159] {
        for vad in MODES {
            // From the middle of the utterance, so the samples are not silent.
            let samples = &speech[20_000..20_000 + len];
            let got = whistle().transcribe(samples, &options(vad)).unwrap();
            assert_eq!(
                (got.text.as_str(), got.language.as_deref(), got.tokens.len()),
                ("", None, 0),
                "{len} samples, {vad:?}"
            );
        }
    }
    // One hop or more has a valid frame and decodes (what it says is not pinned).
    for len in [160, 320] {
        let samples = &speech[20_000..20_000 + len];
        assert!(whistle()
            .transcribe(samples, &options(VadMode::Off))
            .is_ok());
    }
}

#[test]
fn an_unreadable_wav_file_is_an_audio_error() {
    let err = whistle()
        .transcribe_wav(b"not a wav file", &Options::default())
        .unwrap_err();
    assert_eq!(err, Error::Audio(AudioError::NotWav));
    assert_eq!(err.to_string(), AudioError::NotWav.to_string());
    assert!(err.source().is_some());
}

#[test]
fn a_wav_over_thirty_seconds_at_44_1_khz_is_rejected_with_the_gate_off() {
    let bytes = wav_bytes(&vec![0.0; 31 * 44_100], 44_100);
    for vad in [VadMode::Off, VadMode::Gate] {
        let err = whistle().transcribe_wav(&bytes, &options(vad)).unwrap_err();
        assert!(
            matches!(&err, Error::Input(m) if m.contains("30 s")),
            "{err}"
        );
    }
}

#[test]
fn input_over_ten_minutes_is_rejected_before_trimming() {
    let long = vec![0.0f32; MAX_INPUT_SAMPLES + 1];
    let err = whistle()
        .transcribe(&long, &Options::default())
        .unwrap_err();
    assert!(
        matches!(&err, Error::Input(m) if m.contains("600 s")),
        "{err}"
    );
}
