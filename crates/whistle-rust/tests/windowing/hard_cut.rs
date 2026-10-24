// The hard-cut fallback: speech with no pause to cut in. Two inputs: the English clips joined
// with no gaps (about 35 s) with the detector off, where the cut is fixed at 29 s; and the
// clips' speech alone (each clip's detected span without its pads) joined into one 33 s run the
// detector sees as a single segment, so with `GateAndTrim` the cut goes at the quietest frame
// before the limit. Each window overlaps the cut by `OVERLAP_SAMPLES` and keeps the words whose
// midpoint is on its side. The reference for "nothing doubled or lost" is what the two windows
// heard in the overlap on their own: every word both heard there (their longest common
// subsequence) is in the stitch, the stitch has as many words there as the windows heard
// (between the smaller and the larger count), and it keeps most of each window's own words
// there (recall, bound by measurement). Also: every window is at most 30 s, and text and
// tokens are the same with word times on and off.

use whistle::{plan_windows, VadMode, Word, MAX_SAMPLES, OVERLAP_SAMPLES};

use crate::shared::{alone, clip, concat, greedy, lcs, seconds, whistle, ENGLISH};
use crate::wer::normalize;

/// The lowest share of a window's own overlap words the stitch keeps (the better of the two
/// windows): measured 0.86 at the fixed cut and 0.83 at the detector's cut (2026-10-24, the
/// windows disagree on the cut's last word); one more word lost takes a 6- or 7-word overlap
/// to 0.71 at best.
const MIN_OVERLAP_RECALL: f64 = 0.8;

/// About 35 s of the English clips in turn, no gaps.
fn no_gaps() -> Vec<f32> {
    let clips: Vec<(&str, f32)> = (0..11).map(|i| (ENGLISH[i % 5], 0.0)).collect();
    concat(&clips).0
}

/// The clips' speech only (each clip's detected span less its 0.1 s and 0.2 s pads), 12 clips
/// joined into one segment of about 33 s.
fn one_speech_run() -> Vec<f32> {
    let mut out = Vec::new();
    for i in 0..12 {
        let c = clip(ENGLISH[i % 5]);
        let segments = whistle::vad::speech_segments(&c);
        let (first, last) = (segments[0].start, segments[segments.len() - 1].end);
        out.extend_from_slice(&c[first + 1600..last - 3200]);
    }
    out
}

/// The words of `text` (normalized), as one list.
fn words_of(words: &[Word]) -> Vec<String> {
    let text: Vec<&str> = words.iter().map(|w| w.word.as_str()).collect();
    normalize(&text.join(" "))
}

fn check_hard_cuts(samples: &[f32], vad: VadMode) {
    let windows = plan_windows(samples, vad);
    assert!(windows.len() >= 2, "{vad:?}: {windows:?}");
    for w in &windows {
        assert!(w.decode.len() <= MAX_SAMPLES, "{vad:?}: {w:?}");
    }
    let options = greedy(vad, true);
    let got = whistle().transcribe(samples, &options).unwrap();
    let untimed = whistle().transcribe(samples, &greedy(vad, false)).unwrap();
    assert_eq!(got.text, untimed.text, "{vad:?}");
    assert_eq!(got.tokens, untimed.tokens, "{vad:?}");
    let parts: Vec<Vec<Word>> = windows
        .iter()
        .map(|w| {
            let offset = seconds(w.decode.start);
            alone(samples, w, &options)
                .words
                .into_iter()
                .map(|mut word| {
                    word.start += offset;
                    word.end += offset;
                    word
                })
                .collect()
        })
        .collect();
    for (k, pair) in windows.windows(2).enumerate() {
        let Some(cut) = pair[0].keep_until else {
            continue;
        };
        assert_eq!(pair[1].keep_from, Some(cut), "{vad:?}");
        let overlap = seconds(cut - OVERLAP_SAMPLES)..seconds(cut + OVERLAP_SAMPLES);
        let within = |words: &[Word]| -> Vec<Word> {
            words
                .iter()
                .filter(|w| overlap.contains(&((w.start + w.end) / 2.0)))
                .cloned()
                .collect()
        };
        let s = words_of(&within(&got.words));
        let b = words_of(&within(&parts[k]));
        let a = words_of(&within(&parts[k + 1]));
        let recall = |r: &[String]| lcs(r, &s) as f64 / r.len().max(1) as f64;
        let (rb, ra) = (recall(&b), recall(&a));
        let heard = format!(
            "{vad:?} cut {:.2} s: stitched {s:?}, before {b:?} ({rb:.2}), after {a:?} ({ra:.2})",
            seconds(cut)
        );
        println!("{heard}");
        // Every word both windows heard there is in the stitch, in order.
        let agreed = agreed_words(&b, &a);
        assert_eq!(
            lcs(&agreed, &s),
            agreed.len(),
            "a word both windows heard is lost: {heard}"
        );
        assert!(
            (b.len().min(a.len())..=b.len().max(a.len())).contains(&s.len()),
            "a word doubled or lost: {heard}"
        );
        assert!(rb.max(ra) >= MIN_OVERLAP_RECALL, "{heard}");
    }
    for pair in got.words.windows(2) {
        assert!(pair[0].end <= pair[1].start, "{vad:?} {pair:?}");
    }
}

/// A longest common subsequence of `a` and `b`, as words.
fn agreed_words(a: &[String], b: &[String]) -> Vec<String> {
    let matched = crate::shared::lcs_matches(a, b);
    a.iter()
        .zip(matched)
        .filter(|(_, m)| *m)
        .map(|(w, _)| w.clone())
        .collect()
}

#[test]
fn speech_without_a_pause_is_hard_cut_at_fixed_places_with_the_detector_off() {
    check_hard_cuts(&no_gaps(), VadMode::Off);
}

#[test]
fn one_speech_run_over_30_s_is_hard_cut_at_its_quietest_frame() {
    let samples = one_speech_run();
    let segments = whistle::vad::speech_segments(&samples);
    assert_eq!(segments.len(), 1, "{segments:?}");
    assert!(segments[0].len() > MAX_SAMPLES, "{segments:?}");
    let windows = plan_windows(&samples, VadMode::GateAndTrim);
    assert!(windows[0].keep_until.is_some(), "{windows:?}");
    check_hard_cuts(&samples, VadMode::GateAndTrim);
}
