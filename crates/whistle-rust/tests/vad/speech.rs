// Speech must be found: every speech clip clean, under white noise at 10 dB SNR, 30 dB quieter,
// padded by a second of silence each side, and padded by 2 s of white noise each side at 10 and
// 20 dB SNR (trimming must work on noise, not only on digital silence). Segments must cover at
// least 90% of the engine's word span, reach no more than 0.5 s outside it, and sit more than
// `merge_gap_s` apart.

use std::ops::Range;

use whistle::vad::{has_speech, speech_segments, VadConfig};

use crate::signals::{covered, rms, speech_clips, white, RATE};

const MIN_COVERAGE: f64 = 0.9;
const MAX_OVERHANG: usize = RATE / 2;

/// Checks `segments` against the word span; returns a failure description, if any.
pub fn span_failure(segments: &[Range<usize>], words: &Range<usize>) -> Option<String> {
    let coverage = covered(segments, words) as f64 / (words.end - words.start) as f64;
    let first = segments.first().map_or(usize::MAX, |s| s.start);
    let last = segments.last().map_or(0, |s| s.end);
    let early = words.start.saturating_sub(first);
    let late = last.saturating_sub(words.end);
    if segments.is_empty() || coverage < MIN_COVERAGE || early > MAX_OVERHANG || late > MAX_OVERHANG
    {
        Some(format!(
            "coverage {coverage:.3}, starts {early} samples early, ends {late} late: \
             {segments:?} for words {words:?}"
        ))
    } else {
        None
    }
}

/// A failure description when two segments are closer than the default `merge_gap_s`.
pub fn spacing_failure(segments: &[Range<usize>]) -> Option<String> {
    let gap = (VadConfig::default().merge_gap_s * RATE as f32) as usize;
    segments
        .windows(2)
        .find(|p| p[1].start - p[0].end <= gap)
        .map(|p| format!("segments {:?} and {:?} closer than merge_gap_s", p[0], p[1]))
}

fn check_all(variant: &str, make: impl Fn(&[f32]) -> Vec<f32>) {
    let mut failures = Vec::new();
    for clip in speech_clips() {
        let samples = make(&clip.samples);
        let segments = speech_segments(&samples);
        if let Some(why) = spacing_failure(&segments) {
            failures.push(format!("{}: {why}", clip.name));
        }
        if !has_speech(&samples) {
            failures.push(format!("{}: no speech", clip.name));
        }
        if let Some(why) = span_failure(&segments, &clip.words) {
            failures.push(format!("{}: {why}", clip.name));
        }
    }
    assert!(failures.is_empty(), "{variant}: {failures:#?}");
}

/// `samples` with white noise at `snr_db` below the speech's RMS over the whole clip.
pub fn with_noise(samples: &[f32], snr_db: f32, seed: u64) -> Vec<f32> {
    let level = 20.0 * rms(samples).log10() - snr_db;
    let noise = white(samples.len() as f32 / RATE as f32 + 1.0, level, seed);
    samples.iter().zip(&noise).map(|(s, n)| s + n).collect()
}

pub fn attenuated(samples: &[f32], db: f32) -> Vec<f32> {
    let gain = 10f32.powf(-db / 20.0);
    samples.iter().map(|s| s * gain).collect()
}

#[test]
fn every_speech_clip_is_speech_across_its_words() {
    check_all("clean", <[f32]>::to_vec);
}

#[test]
fn speech_under_white_noise_at_10_db_snr_is_speech() {
    check_all("10 dB SNR", |s| with_noise(s, 10.0, 7));
}

#[test]
fn a_quiet_talker_30_db_down_is_speech() {
    check_all("-30 dB", |s| attenuated(s, 30.0));
}

#[test]
fn a_second_of_silence_each_side_is_mostly_excluded() {
    let mut failures = Vec::new();
    for clip in speech_clips() {
        let mut samples = vec![0.0f32; RATE];
        samples.extend(&clip.samples);
        samples.extend(vec![0.0f32; RATE]);
        let segments = speech_segments(&samples);
        let words = clip.words.start + RATE..clip.words.end + RATE;
        if let Some(why) = span_failure(&segments, &words) {
            failures.push(format!("{}: {why}", clip.name));
        }
        let pad_before = covered(&segments, &(0..RATE));
        let pad_after = covered(&segments, &(samples.len() - RATE..samples.len()));
        // "Most" excluded: at most a third of either second of padding.
        if pad_before > RATE / 3 || pad_after > RATE / 3 {
            failures.push(format!(
                "{}: padding covered {pad_before} before, {pad_after} after",
                clip.name
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// `clip` with `pad` samples of white noise before and after, the noise running under the speech
/// too, `snr_db` below the speech's RMS.
fn noise_padded(clip: &[f32], pad: usize, snr_db: f32, seed: u64) -> Vec<f32> {
    let level = 20.0 * rms(clip).log10() - snr_db;
    let mut samples = white(
        (clip.len() + 2 * pad) as f32 / RATE as f32 + 1.0,
        level,
        seed,
    );
    samples.truncate(clip.len() + 2 * pad);
    for (s, c) in samples[pad..].iter_mut().zip(clip) {
        *s += c;
    }
    samples
}

fn check_noise_padded(snr_db: f32) {
    let pad = 2 * RATE;
    let mut failures = Vec::new();
    for clip in speech_clips() {
        let samples = noise_padded(&clip.samples, pad, snr_db, 13);
        let segments = speech_segments(&samples);
        let words = clip.words.start + pad..clip.words.end + pad;
        if let Some(why) = span_failure(&segments, &words) {
            failures.push(format!("{}: {why}", clip.name));
        }
        if let Some(why) = spacing_failure(&segments) {
            failures.push(format!("{}: {why}", clip.name));
        }
        let pad_before = covered(&segments, &(0..pad));
        let pad_after = covered(&segments, &(samples.len() - pad..samples.len()));
        // "Most" of each 2 s noise pad excluded: at most a quarter of it covered.
        if pad_before > pad / 4 || pad_after > pad / 4 {
            failures.push(format!(
                "{}: noise pad covered {pad_before} before, {pad_after} after",
                clip.name
            ));
        }
    }
    assert!(failures.is_empty(), "{snr_db} dB SNR: {failures:#?}");
}

#[test]
fn two_seconds_of_noise_each_side_at_10_db_snr_are_trimmed() {
    check_noise_padded(10.0);
}

#[test]
fn two_seconds_of_noise_each_side_at_20_db_snr_are_trimmed() {
    check_noise_padded(20.0);
}
