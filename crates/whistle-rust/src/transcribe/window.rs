//! Windowing (`docs/design/windowing.md`): audio longer than the encoder's 30 s training window
//! is split into windows of at most [`MAX_SAMPLES`], each decoded alone by
//! `Whistle::decode_window` and stitched by `long`. This module only plans the cuts. With the
//! voice-activity detector on, windows are packed from its speech segments and cut in the
//! middle of a pause; a stretch of speech with no usable pause, and every cut with the detector
//! off, is a hard cut: the two windows overlap by [`OVERLAP_SAMPLES`] each side of the cut, and
//! each keeps only the words whose midpoint lies on its side.

use std::ops::Range;

use super::{VadMode, MAX_SAMPLES};
use crate::audio::SAMPLE_RATE;
use crate::vad;

/// Samples each window decodes past a hard cut (1 s): longer than almost any single word, so
/// a word the cut splits is whole in at least one of the two windows.
pub const OVERLAP_SAMPLES: usize = SAMPLE_RATE as usize;

/// How far back from the latest possible hard cut the quietest frame is searched for (4 s).
const HARD_CUT_SEARCH: usize = 4 * SAMPLE_RATE as usize;

/// The hard-cut search's frame: one encoder frame, 80 ms.
const CUT_FRAME: usize = 1280;

/// One window of a long input, in 16 kHz samples of the input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Window {
    /// The samples decoded, at most [`MAX_SAMPLES`] long.
    pub decode: Range<usize>,
    /// A hard cut before this window: its words whose midpoint lies before this sample belong
    /// to the previous window and are dropped.
    pub keep_from: Option<usize>,
    /// A hard cut after this window: its words whose midpoint lies at or after this sample
    /// belong to the next window and are dropped.
    pub keep_until: Option<usize>,
}

/// The windows `Whistle::transcribe` decodes for 16 kHz mono `samples` under `vad`: empty when
/// the gate finds no speech (or there are no samples); one window, exactly the span a single
/// pass decodes, when that span fits in [`MAX_SAMPLES`]; otherwise windows as
/// `docs/design/windowing.md` describes.
pub fn plan_windows(samples: &[f32], vad: VadMode) -> Vec<Window> {
    match vad {
        VadMode::Off => plan(samples, None, false),
        mode => {
            let segments = vad::speech_segments(samples);
            plan(samples, Some(&segments), mode == VadMode::GateAndTrim)
        }
    }
}

/// [`plan_windows`] given the detector's `segments` (`None` with it off; empty means no speech).
/// `trim` decodes only each window's speech, as [`VadMode::GateAndTrim`] does for one pass.
pub(crate) fn plan(samples: &[f32], segments: Option<&[Range<usize>]>, trim: bool) -> Vec<Window> {
    let region = match segments {
        None => 0..samples.len(),
        Some(s) => match (s.first(), s.last()) {
            (Some(first), Some(last)) if trim => first.start..last.end,
            (Some(_), Some(_)) => 0..samples.len(),
            _ => return Vec::new(),
        },
    };
    if region.is_empty() {
        return Vec::new();
    }
    let segments = segments.unwrap_or(&[]);
    let mut windows = Vec::new();
    // `start`: where the next window's span begins; `keep`: the hard cut it begins at, if any
    // (then `start` is that cut minus the overlap).
    let mut start = region.start;
    let mut keep = None;
    loop {
        let limit = start + MAX_SAMPLES;
        if region.end <= limit {
            windows.push(window(segments, trim, start..region.end, keep, None));
            return windows;
        }
        let gap = if segments.is_empty() {
            None
        } else {
            best_gap(segments, start, limit)
        };
        match gap {
            Some(Pause { cut, end, middle }) => {
                windows.push(window(segments, trim, start..cut, keep, None));
                // The next window's 30 s count from the speech after the pause, unless `Gate`
                // tiles a pause cut in its middle (each half to the window beside it).
                start = if middle && !trim { cut } else { end };
                keep = None;
            }
            None => {
                let latest = limit - OVERLAP_SAMPLES;
                let cut = if segments.is_empty() {
                    // The detector off: fixed cuts, as the brief's "no VAD" asks.
                    latest
                } else {
                    quietest_frame(samples, latest.saturating_sub(HARD_CUT_SEARCH), latest)
                };
                let span = start..cut + OVERLAP_SAMPLES;
                windows.push(window(segments, trim, span, keep, Some(cut)));
                start = cut - OVERLAP_SAMPLES;
                keep = Some(cut);
            }
        }
    }
}

/// The window over `span`: with `trim`, narrowed to the speech it holds on the sides that are
/// not hard cuts.
fn window(
    segments: &[Range<usize>],
    trim: bool,
    span: Range<usize>,
    keep_from: Option<usize>,
    keep_until: Option<usize>,
) -> Window {
    let mut decode = span.clone();
    if trim {
        let inside = || {
            segments
                .iter()
                .filter(|s| s.end > span.start && s.start < span.end)
        };
        if keep_from.is_none() {
            if let Some(first) = inside().next() {
                decode.start = first.start.max(span.start);
            }
        }
        if keep_until.is_none() {
            if let Some(last) = inside().next_back() {
                decode.end = last.end.min(span.end);
            }
        }
    }
    Window {
        decode,
        keep_from,
        keep_until,
    }
}

/// A pause a window is cut in.
struct Pause {
    /// Where the window ends.
    cut: usize,
    /// Where the pause ends (the next speech segment's start).
    end: usize,
    /// Whether `cut` is the pause's middle (else the pause crossed the window's limit).
    middle: bool,
}

/// The cut for a window starting at `start`: in the longest pause between two speech segments
/// that starts after `start` and before `limit`, preferring pauses whose cut leaves the window
/// at least half full (so one early pause does not make a run of short windows); the later
/// pause on a tie. The cut is the pause's middle, or `limit` when the middle is past it: a
/// pause that crosses the limit is still a pause, and cutting in it beats a hard cut inside
/// speech. `None` when no pause starts in reach.
fn best_gap(segments: &[Range<usize>], start: usize, limit: usize) -> Option<Pause> {
    let half = start + MAX_SAMPLES / 2;
    let mut best: Option<((bool, usize, usize), Pause)> = None;
    for pair in segments.windows(2) {
        let (gap_start, gap_end) = (pair[0].end, pair[1].start);
        if gap_start < start || gap_end <= gap_start {
            continue;
        }
        if gap_start >= limit {
            break;
        }
        let middle = gap_start + (gap_end - gap_start) / 2;
        let cut = middle.min(limit);
        if cut <= start {
            continue;
        }
        let key = (cut >= half, gap_end - gap_start, cut);
        if best.as_ref().is_none_or(|(b, _)| key >= *b) {
            let pause = Pause {
                cut,
                end: gap_end,
                middle: cut == middle,
            };
            best = Some((key, pause));
        }
    }
    best.map(|(_, pause)| pause)
}

/// The middle of the lowest-energy [`CUT_FRAME`] in `lo..hi` (frames stepped back from `hi`),
/// the latest on a tie; `hi` when the range holds no whole frame.
fn quietest_frame(samples: &[f32], lo: usize, hi: usize) -> usize {
    let mut best: Option<(f64, usize)> = None;
    let mut end = hi.min(samples.len());
    while end >= lo + CUT_FRAME {
        let energy: f64 = samples[end - CUT_FRAME..end]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        if best.is_none_or(|(e, _)| energy < e) {
            best = Some((energy, end - CUT_FRAME / 2));
        }
        end -= CUT_FRAME;
    }
    best.map_or(hi, |(_, cut)| cut)
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: usize = SAMPLE_RATE as usize;

    #[test]
    fn audio_that_fits_is_one_window_over_the_single_pass_span() {
        let segs = [S..2 * S, 5 * S..9 * S];
        let w = plan(&vec![0.0; 20 * S], Some(&segs), true);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].decode, S..9 * S);
        let w = plan(&vec![0.0; 20 * S], Some(&segs), false);
        assert_eq!(w[0].decode, 0..20 * S);
        assert!(plan(&[0.0; 100], Some(&[]), true).is_empty());
    }

    #[test]
    fn long_speech_is_cut_in_the_middle_of_the_longest_late_pause() {
        // Segments every 5 s with 1 s pauses; the pause at 23.5-25 s is the longest that fits.
        let segs: Vec<_> = (0..12)
            .map(|i| {
                let end = if i == 4 {
                    47 * S / 2
                } else {
                    i * 5 * S + 4 * S
                };
                i * 5 * S..end
            })
            .collect();
        let samples = vec![0.0; 60 * S];
        let w = plan(&samples, Some(&segs), true);
        assert_eq!(w[0].decode, 0..47 * S / 2);
        assert_eq!(w[1].decode.start, 25 * S);
        for win in &w {
            assert!(win.decode.len() <= MAX_SAMPLES, "{win:?}");
            assert_eq!((win.keep_from, win.keep_until), (None, None));
        }
        assert_eq!(w.last().unwrap().decode.end, 59 * S);
    }

    #[test]
    fn a_pause_crossing_the_limit_is_cut_at_the_limit_and_skipped() {
        // 1-17 s of speech, a 26 s pause (its middle at exactly 30 s), 43-60 s of speech.
        let segs = [S..17 * S, 43 * S..60 * S];
        let w = plan(&vec![0.0; 61 * S], Some(&segs), true);
        assert_eq!(w.len(), 2, "{w:?}");
        assert_eq!(w[0].decode, S..17 * S);
        assert_eq!(w[1].decode, 43 * S..60 * S);
        // Without the trim, a pause cut in its middle is shared by the windows beside it.
        let w = plan(&vec![0.0; 60 * S], Some(&segs), false);
        assert_eq!(w.len(), 2, "{w:?}");
        assert_eq!(w[0].decode, 0..30 * S);
        assert_eq!(w[1].decode, 30 * S..60 * S);
        for win in &w {
            assert_eq!((win.keep_from, win.keep_until), (None, None), "{win:?}");
        }
        // A pause that crosses the limit: the rest of it is never decoded.
        let segs = [S..17 * S, 80 * S..90 * S];
        let w = plan(&vec![0.0; 91 * S], Some(&segs), false);
        assert_eq!(w.len(), 2, "{w:?}");
        assert_eq!(w[0].decode, 0..30 * S);
        assert_eq!(w[1].decode, 80 * S..91 * S);
    }

    #[test]
    fn speech_without_a_pause_is_hard_cut_with_an_overlap() {
        let mut samples = vec![0.5f32; 70 * S];
        // A quiet frame inside the search range.
        samples[27 * S..27 * S + CUT_FRAME].fill(0.0);
        let segs = [Range {
            start: 0,
            end: 70 * S,
        }];
        let w = plan(&samples, Some(&segs), true);
        let cut = w[0].keep_until.unwrap();
        assert!((27 * S..27 * S + CUT_FRAME).contains(&cut), "{cut}");
        assert_eq!(w[0].decode.end, cut + OVERLAP_SAMPLES);
        assert_eq!(w[1].decode.start, cut - OVERLAP_SAMPLES);
        assert_eq!(w[1].keep_from, Some(cut));
        for win in &w {
            assert!(win.decode.len() <= MAX_SAMPLES, "{win:?}");
        }
        assert_eq!(w.last().unwrap().decode.end, 70 * S);
    }

    #[test]
    fn with_the_detector_off_cuts_are_fixed() {
        let w = plan(&vec![0.0; 70 * S], None, false);
        let cuts: Vec<_> = w.iter().filter_map(|w| w.keep_until).collect();
        assert_eq!(cuts, vec![29 * S, 57 * S]);
        assert_eq!(w[0].decode, 0..30 * S);
        assert_eq!(w[1].decode, 28 * S..58 * S);
        assert_eq!(w[2].decode, 56 * S..70 * S);
    }
}
