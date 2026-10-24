//! From frame measurements (`features`) to speech segments: per-band noise floors, the
//! syllable-rate modulation test, the hysteresis decision, and the run clean-up (merge short
//! gaps, drop short bursts, pad by the hangover, merge again). Every threshold comes from
//! `VadConfig` (`vad/mod.rs`).

use std::ops::Range;

use super::features::{Frame, BANDS, WIN};
use super::VadConfig;
use crate::audio::{HOP, SAMPLE_RATE};

/// Seconds to whole frames (hops): at least `min`, at most `cap` (the input's frame count, so
/// window arithmetic and buffers stay bounded by the input whatever the configuration says).
fn frames_for(seconds: f32, min: usize, cap: usize) -> usize {
    // Negative and NaN seconds count as 0; the cast saturates, so +∞ reaches `cap`.
    let frames = if seconds > 0.0 {
        (seconds * SAMPLE_RATE as f32 / HOP as f32).round() as usize
    } else {
        0
    };
    frames.max(min).min(cap)
}

/// Seconds to samples, at most `cap` (the input's length).
fn samples_for(seconds: f32, cap: usize) -> usize {
    // Negative and NaN seconds count as 0; the cast saturates, so +∞ reaches `cap`.
    let samples = if seconds > 0.0 {
        (seconds * SAMPLE_RATE as f32).round() as usize
    } else {
        0
    };
    samples.min(cap)
}

/// The minimum of `values` over the window `t - reach ..= t + reach` (cut at the ends) for every
/// `t`, by a monotonic deque: each value enters and leaves the deque once, so the cost does not
/// grow with the window.
fn sliding_min(values: &[f32], reach: usize) -> Vec<f32> {
    let n = values.len();
    let mut deque = std::collections::VecDeque::with_capacity((2 * reach + 1).min(n));
    let mut next = 0;
    (0..n)
        .map(|t| {
            while next < n && next <= t + reach {
                // Strictly greater values can never be a later window's minimum once `next` is in.
                while deque
                    .back()
                    .is_some_and(|&b: &usize| values[b] > values[next])
                {
                    deque.pop_back();
                }
                deque.push_back(next);
                next += 1;
            }
            while deque.front().is_some_and(|&f| f + reach < t) {
                deque.pop_front();
            }
            deque.front().map_or(f32::INFINITY, |&f| values[f])
        })
        .collect()
}

/// How far below `absolute_floor_db` a single band's floor may go. One band of six can hold a
/// hundredth of a frame's power and still be where its formant is, so the band floors bottom out
/// 20 dB lower; the frame's total power is what the absolute floor itself gates.
const BAND_FLOOR_BELOW_ABSOLUTE_DB: f32 = 20.0;

/// A band more than this far below the frame's loudest band does not count towards the frame's
/// rise. Such a band holds only leakage from its neighbours (a tone's window sidelobes, say),
/// whose tiny level wobbles by many dB without any sound behind it; a speech frame's formant and
/// fricative bands sit well within this of each other.
const BAND_RELEVANCE_DB: f32 = 30.0;

/// Frames each side averaged into the envelope whose modulation is measured: ±2 frames, a 50 ms
/// moving average. Syllables (2–8 Hz, 125–500 ms apart) pass through it almost untouched, while
/// the frame-to-frame scatter of noise, which would otherwise read as a dB or two of modulation,
/// is averaged down.
const ENVELOPE_REACH: usize = 2;

/// The noise floor under each frame in each band, in dB: the minimum of the lightly smoothed
/// band energy over a window centred on the frame, never below the band's absolute floor.
/// Speech pauses between words and syllables, so within a couple of seconds its quietest stretch
/// is the background; steady noise sits at its own floor everywhere, so it never clears the
/// margin.
fn band_floor(frames: &[Frame], band: usize, config: &VadConfig) -> Vec<f32> {
    let n = frames.len();
    let half_smooth = frames_for(config.floor_smoothing_s, 1, n) / 2;
    let powers: Vec<f64> = frames
        .iter()
        .map(|f| 10f64.powf(f64::from(f.band_db[band]) / 10.0))
        .collect();
    let smoothed: Vec<f32> = (0..n)
        .map(|t| {
            let span = &powers[t.saturating_sub(half_smooth)..(t + half_smooth + 1).min(n)];
            // Averaging in the power domain keeps one quiet frame from dragging the floor down
            // further than its share.
            let power = span.iter().sum::<f64>() / span.len() as f64;
            (10.0 * power.log10()) as f32
        })
        .collect();
    let reach = frames_for(config.floor_window_s, 1, n);
    let lowest = config.absolute_floor_db - BAND_FLOOR_BELOW_ABSOLUTE_DB;
    sliding_min(&smoothed, reach)
        .into_iter()
        .map(|min| min.max(lowest))
        .collect()
}

/// A frame's total power above the low cut, in dB.
fn total_db(frame: &Frame) -> f32 {
    let total: f64 = frame
        .band_db
        .iter()
        .map(|&db| 10f64.powf(f64::from(db) / 10.0))
        .sum();
    (10.0 * total.log10()) as f32
}

/// Each frame's level above the background, in dB: the largest rise of any band over its own
/// floor. Speech concentrates its power in a formant band at a time, so the best band rises far
/// more than the frame's total power does under broadband noise.
pub(crate) fn rise(frames: &[Frame], config: &VadConfig) -> Vec<f32> {
    let floors: Vec<Vec<f32>> = (0..BANDS).map(|b| band_floor(frames, b, config)).collect();
    frames
        .iter()
        .enumerate()
        .map(|(t, f)| {
            // Below the absolute floor nothing rises: digital silence and near-silence are
            // never speech, whatever the band floors do.
            if total_db(f) < config.absolute_floor_db {
                return f32::NEG_INFINITY;
            }
            let loudest = f.band_db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (0..BANDS)
                .filter(|&b| f.band_db[b] >= loudest - BAND_RELEVANCE_DB)
                .map(|b| f.band_db[b] - floors[b][t])
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect()
}

/// The modulation under each frame, in dB: the interquartile range of the energy envelope (each
/// frame's total power, smoothed) over the frames near it whose rise clears `margin_off_db`. It
/// is measured on the energy itself, not on the rise, so the step the sliding-minimum floor
/// takes when old silence leaves its window cannot pose as modulation. Speech rises and falls by
/// tens of dB at syllable rate; a hum, a tone, a fan or a hiss that switches on stays within a
/// dB or two of itself. Only loud
/// frames count, and the interquartile range ignores the few frames straddling an onset, so a
/// steady sound switching on is not mistaken for a syllable. Fewer than `min_modulation_span_s`
/// of loud frames nearby (a click) gives 0.
pub(crate) fn modulation(frames: &[Frame], rise: &[f32], config: &VadConfig) -> Vec<f32> {
    let n = rise.len();
    let total: Vec<f32> = frames.iter().map(total_db).collect();
    let reach = frames_for(config.modulation_window_s / 2.0, 1, n);
    let min_active = frames_for(config.min_modulation_span_s, 2, n.max(2));
    let envelope: Vec<f32> = (0..n)
        .map(|t| {
            let span = &total[t.saturating_sub(ENVELOPE_REACH)..(t + ENVELOPE_REACH + 1).min(n)];
            span.iter().sum::<f32>() / span.len() as f32
        })
        .collect();
    let mut loud = Vec::with_capacity((2 * reach + 1).min(n));
    (0..n)
        .map(|t| {
            loud.clear();
            let (lo, hi) = (t.saturating_sub(reach), (t + reach + 1).min(n));
            loud.extend(
                (lo..hi)
                    .filter(|&u| rise[u] > config.margin_off_db)
                    .map(|u| envelope[u]),
            );
            if loud.len() < min_active {
                return 0.0;
            }
            loud.sort_by(f32::total_cmp);
            let quartile = |q: usize| loud[(q * (loud.len() - 1)) / 4];
            quartile(3) - quartile(1)
        })
        .collect()
}

/// The per-frame decision with hysteresis: a run starts on a frame `margin_on_db` above the
/// floor and continues while frames stay `margin_off_db` above it; every speech frame must also
/// be modulated like speech.
fn speech_frames(rise: &[f32], modulation: &[f32], c: &VadConfig) -> Vec<bool> {
    let mut active = false;
    rise.iter()
        .zip(modulation)
        .map(|(&rise, &m)| {
            let speech_like = m >= c.min_modulation_db;
            let margin = if active {
                c.margin_off_db
            } else {
                c.margin_on_db
            };
            active = speech_like && rise > margin;
            active
        })
        .collect()
}

/// Runs of speech frames, as frame ranges, with gaps up to `merge_gap_s` bridged and merged
/// runs shorter than `min_speech_s` dropped.
fn frame_runs(speech: &[bool], len: usize, c: &VadConfig) -> Vec<Range<usize>> {
    let mut runs: Vec<Range<usize>> = Vec::new();
    let gap = frames_for(c.merge_gap_s, 0, speech.len());
    let mut t = 0;
    while t < speech.len() {
        if !speech[t] {
            t += 1;
            continue;
        }
        let start = t;
        while t < speech.len() && speech[t] {
            t += 1;
        }
        match runs.last_mut() {
            Some(last) if start - last.end <= gap => last.end = t,
            _ => runs.push(start..t),
        }
    }
    // Capped one past the input so a huge minimum drops every run rather than overflowing.
    let min = samples_for(c.min_speech_s, len.saturating_add(1));
    runs.retain(|r| (r.end - r.start - 1) * HOP + WIN >= min);
    runs
}

/// The full decision: frame measurements to padded, merged sample ranges within `len`.
pub(crate) fn segments(frames: &[Frame], len: usize, c: &VadConfig) -> Vec<Range<usize>> {
    let rise = rise(frames, c);
    let modulation = modulation(frames, &rise, c);
    let speech = speech_frames(&rise, &modulation, c);
    let before = samples_for(c.hangover_before_s, len);
    let after = samples_for(c.hangover_after_s, len);
    let gap = samples_for(c.merge_gap_s, len);
    let mut out: Vec<Range<usize>> = Vec::new();
    for run in frame_runs(&speech, len, c) {
        let start = (run.start * HOP).saturating_sub(before);
        let end = ((run.end - 1) * HOP + WIN + after).min(len);
        // Merging again after the padding keeps every pair of segments a caller sees at least
        // `merge_gap_s` apart, as the padding can bring runs closer than the frame merge saw.
        match out.last_mut() {
            Some(last) if start <= last.end + gap => last.end = last.end.max(end),
            _ => out.push(start..end),
        }
    }
    out
}
