// Known limitations, pinned so a change that fixes or worsens them is visible: sounds that are
// neither speech nor steady (intermittent beeps, a typing-like click train, noise modulated at a
// syllable rate), and that the detector currently calls speech (docs/design/vad.md
// "Hard to separate"). When one of these starts passing as non-speech, flip its expectation.

use whistle::vad::speech_segments;

use crate::signals::{tone, white, RATE};

/// 1 kHz at −20 dBFS, 150 ms on and 150 ms off, for 3 s, over an optional noise bed.
fn beeps(bed_dbfs: Option<f32>) -> Vec<f32> {
    let mut samples = tone(1000.0, 3.0, -20.0);
    let period = (0.3 * RATE as f32) as usize;
    for (i, s) in samples.iter_mut().enumerate() {
        if i % period >= period / 2 {
            *s = 0.0;
        }
    }
    if let Some(level) = bed_dbfs {
        for (s, n) in samples.iter_mut().zip(white(3.0, level, 51)) {
            *s += n;
        }
    }
    samples
}

/// A typing-like click train: about 8 short noise bursts (5 ms) per second for 3 s.
fn click_train() -> Vec<f32> {
    let mut samples = vec![0.0f32; 3 * RATE];
    let burst = white(0.005, -10.0, 61);
    let spacing = RATE / 8;
    for k in 0..samples.len() / spacing {
        // Uneven spacing, as keystrokes are.
        let at = k * spacing + (k * 397) % (spacing / 3);
        for (s, b) in samples[at..].iter_mut().zip(&burst) {
            *s += b;
        }
    }
    samples
}

/// White noise at −20 dBFS whose amplitude swings fully at 4 Hz (a syllable-like rate), 3 s.
fn modulated_noise() -> Vec<f32> {
    let mut samples = white(3.0, -20.0, 71);
    for (i, s) in samples.iter_mut().enumerate() {
        let phase = 2.0 * std::f32::consts::PI * 4.0 * i as f32 / RATE as f32;
        *s *= 0.5 - 0.5 * phase.cos();
    }
    samples
}

pub fn named_cases() -> Vec<(&'static str, Vec<f32>)> {
    vec![
        ("beep", beeps(None)),
        ("beep over a -50 dB bed", beeps(Some(-50.0))),
        ("click train", click_train()),
        (
            "noise with full 4 Hz amplitude modulation",
            modulated_noise(),
        ),
    ]
}

fn covered_fraction(samples: &[f32]) -> f64 {
    let covered: usize = speech_segments(samples)
        .iter()
        .map(|s| s.end - s.start)
        .sum();
    covered as f64 / samples.len() as f64
}

#[test]
fn known_limitation_intermittent_beeps_click_trains_and_4_hz_modulated_noise_read_as_speech() {
    // Measured when pinned: both beep cases and the modulated noise are covered completely, the
    // click train 89.5%.
    for (name, samples) in named_cases() {
        let fraction = covered_fraction(&samples);
        assert!(
            fraction > 0.8,
            "{name}: now {fraction:.3} covered; update docs/design/vad.md"
        );
    }
}
