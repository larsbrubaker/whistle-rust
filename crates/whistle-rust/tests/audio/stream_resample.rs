// The streaming resampler (`whistle::audio::StreamResampler`): audio cut into pushes of any
// size comes out bit for bit as the one-shot `resample` makes the whole clip, once `finish`
// has flushed the tail; equal rates pass through; an unsupported rate is an error.

use whistle::audio::{resample, AudioError, StreamResampler};

/// A deterministic signal with content across the band (a chirp plus a few tones).
fn signal(rate: u32, seconds: f64) -> Vec<f32> {
    let n = (f64::from(rate) * seconds) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            let chirp = (2.0 * std::f64::consts::PI * (100.0 + 2000.0 * t) * t).sin();
            let tones = (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                + 0.3 * (2.0 * std::f64::consts::PI * 3100.0 * t).sin();
            (0.3 * chirp + 0.2 * tones) as f32
        })
        .collect()
}

/// `samples` pushed `chunk` at a time, then finished.
fn streamed(samples: &[f32], from: u32, to: u32, chunk: usize) -> Vec<f32> {
    let mut r = StreamResampler::new(from, to).expect("a supported rate");
    let mut out = Vec::new();
    for part in samples.chunks(chunk) {
        out.extend(r.push(part));
    }
    out.extend(r.finish());
    out
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn pushes_of_any_size_give_the_whole_clips_output_exactly() {
    for (from, to) in [
        (44_100, 16_000),
        (48_000, 16_000),
        (22_050, 16_000),
        (8_000, 16_000),
    ] {
        let input = signal(from, 0.7);
        let whole = resample(&input, from, to);
        // 1 sample, an odd prime, a typical device buffer, ~100 ms, and more than the clip.
        for chunk in [1, 37, 512, from as usize / 10, input.len() + 5] {
            let out = streamed(&input, from, to, chunk);
            assert_eq!(out.len(), whole.len(), "{from} -> {to} in {chunk}s");
            assert_eq!(bits(&out), bits(&whole), "{from} -> {to} in {chunk}s");
        }
    }
}

#[test]
fn each_push_returns_what_its_input_completes_and_finish_the_rest() {
    let input = signal(48_000, 0.5);
    let mut r = StreamResampler::new(48_000, 16_000).unwrap();
    let first = r.push(&input[..4800]);
    // Outputs wait for the input half a kernel past them (144 samples at 48 kHz), so 100 ms
    // in gives a little under 100 ms out.
    assert!(first.len() < 1600 && first.len() > 1500, "{}", first.len());
    let rest = r.push(&input[4800..]);
    let tail = r.finish();
    assert!(!tail.is_empty());
    assert_eq!(first.len() + rest.len() + tail.len(), 8000);
    assert!(r.finish().is_empty(), "nothing is owed twice");
}

#[test]
fn equal_rates_pass_through_and_unsupported_rates_are_errors() {
    let input = signal(16_000, 0.1);
    let mut same = StreamResampler::new(16_000, 16_000).unwrap();
    assert_eq!(same.push(&input), input);
    assert!(same.finish().is_empty());
    assert_eq!(
        StreamResampler::new(999, 16_000).err(),
        Some(AudioError::UnsupportedSampleRate(999))
    );
    assert_eq!(
        StreamResampler::new(16_000, 800_000).err(),
        Some(AudioError::UnsupportedSampleRate(800_000))
    );
}
