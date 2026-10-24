// Non-speech must give no speech: digital silence, the empty slice, white and pink noise at three
// levels, mains hum, a steady tone, an isolated click and fan noise (signals::non_speech_cases).

use whistle::vad::{has_speech, speech_segments};

use crate::signals::non_speech_cases;

#[test]
fn silence_steady_noise_tones_and_clicks_have_no_speech() {
    let mut failures = Vec::new();
    for (name, samples) in non_speech_cases() {
        let segments = speech_segments(&samples);
        if !segments.is_empty() || has_speech(&samples) {
            failures.push(format!("{name}: {segments:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "speech found in non-speech: {failures:#?}"
    );
}

#[test]
fn a_tone_switching_on_after_silence_is_not_speech() {
    // The onset is a 50 dB jump; the modulation test must look only at the loud frames, or the
    // switch-on itself would read as a syllable.
    let mut samples = vec![0.0f32; 16000];
    samples.extend(crate::signals::tone(1000.0, 2.0, -20.0));
    samples.extend(vec![0.0f32; 16000]);
    assert_eq!(speech_segments(&samples), vec![]);
}

#[test]
fn white_noise_switching_on_after_silence_is_not_speech() {
    let mut samples = vec![0.0f32; 16000];
    samples.extend(crate::signals::white(2.0, -30.0, 41));
    assert_eq!(speech_segments(&samples), vec![]);
}
