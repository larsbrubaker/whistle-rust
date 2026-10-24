// Long inputs with pauses, with the default trim. A 45 s English input (0.5-1 s of silence
// between clips): every window is at most 30 s, every clip lies inside a window, no cut is a
// hard cut or splits a clip, the transcript is exactly the windows' samples decoded alone (text
// joined with single spaces, tokens concatenated), its words keep the clips' own transcripts
// (word recall, a reference that does not depend on the windows), and each word is its
// window's standalone word offset by the window's start (within one frame), starts inside a
// clip (within two frames), and the words are in order and do not overlap. Pauses longer than
// half a window between two runs of speech that each fit in one window (16 s, whose middle fits
// in the first window, and 25.5 s, which crosses its limit) are cut in, with no hard cut, every
// clip inside a window and no clip skipped (the same recall bound).

use whistle::{plan_windows, Transcript, VadMode, FRAME_SECONDS, MAX_SAMPLES};

use crate::shared::{
    alone, clip_recalls, english_pause_english, every_clip_decoded, greedy, long_english,
    min_recall, seconds, whistle,
};

/// The lowest share of each clip's own words the transcript may keep. It guards against
/// skipped clips, not lost words: a clip the windows skip keeps near 0 of its words (they can
/// only be matched by chance elsewhere), while a clip that is decoded keeps at least 0.57 on
/// these inputs (measured 2026-10-24; its names come out differently beside its neighbours than
/// alone, "Krishtov" for "Krish Taf"). A word or two lost inside a decoded clip stays above
/// the bound; the hard-cut tests check single words at the cut.
const MIN_CLIP_RECALL: f64 = 0.5;

#[test]
fn a_long_input_with_pauses_is_cut_in_them_and_decodes_as_its_windows_alone() {
    let (samples, spans) = long_english(12);
    let options = greedy(VadMode::GateAndTrim, true);
    let windows = plan_windows(&samples, VadMode::GateAndTrim);
    assert!(windows.len() >= 2, "{windows:?}");
    for w in &windows {
        assert!(w.decode.len() <= MAX_SAMPLES, "{w:?}");
        assert_eq!(
            (w.keep_from, w.keep_until),
            (None, None),
            "no hard cut: {w:?}"
        );
    }
    assert!(every_clip_decoded(&windows, &spans), "{windows:?}");
    let got = whistle().transcribe(&samples, &options).unwrap();
    let parts: Vec<Transcript> = windows
        .iter()
        .map(|w| alone(&samples, w, &options))
        .collect();
    let text: Vec<&str> = parts
        .iter()
        .map(|t| t.text.trim())
        .filter(|t| !t.is_empty())
        .collect();
    assert_eq!(got.text, text.join(" "));
    let tokens: Vec<u32> = parts.iter().flat_map(|t| t.tokens.clone()).collect();
    assert_eq!(got.tokens, tokens);
    assert_eq!(got.language.as_deref(), Some("en"));
    let recall = min_recall(
        "45 s input",
        &clip_recalls(&samples, &spans, &options, &got.text),
    );
    assert!(
        recall >= MIN_CLIP_RECALL,
        "recall {recall:.2}: {}",
        got.text
    );

    let want: Vec<_> = windows
        .iter()
        .zip(&parts)
        .flat_map(|(w, part)| {
            let offset = seconds(w.decode.start);
            part.words.iter().cloned().map(move |mut word| {
                word.start += offset;
                word.end += offset;
                word
            })
        })
        .collect();
    assert_eq!(got.words.len(), want.len());
    for (g, w) in got.words.iter().zip(&want) {
        assert_eq!(g.word, w.word);
        assert!((g.start - w.start).abs() <= FRAME_SECONDS, "{g:?} {w:?}");
        assert!((g.end - w.end).abs() <= FRAME_SECONDS, "{g:?} {w:?}");
        // One frame for the window's frame grid, which does not start where the clip does,
        // and one for the alignment's measured precision (docs/design/timestamps.md).
        let slack = 2.0 * FRAME_SECONDS;
        let in_a_clip = spans
            .iter()
            .any(|s| seconds(s.start) - slack <= g.start && g.start <= seconds(s.end));
        assert!(in_a_clip, "{g:?} starts outside every clip");
    }
    for pair in got.words.windows(2) {
        assert!(pair[0].start <= pair[0].end, "{pair:?}");
        assert!(pair[0].end <= pair[1].start, "{pair:?}");
    }
    assert!(got.words.last().unwrap().end <= seconds(samples.len()));
}

#[test]
fn a_pause_longer_than_half_a_window_is_cut_in_and_keeps_every_clip() {
    let options = greedy(VadMode::GateAndTrim, false);
    // 3 clips, 16 s, 2 clips: the pause's middle fits in the first window. 5 clips, 25.5 s,
    // 5 clips (speech 0-17.5 s and 42.7-60.3 s, the review's repro): the pause's middle is at
    // 30.1 s, past the first window's limit, and the second run is long enough that the old
    // planner's second hard cut fell inside it. (With these clip gaps a 25 s pause puts the
    // middle at 29.9 s, inside the limit, so it would not exercise the case.)
    for (before, pause, after) in [(3, 16.0, 2), (5, 25.5, 5)] {
        let (samples, spans) = english_pause_english(before, pause, after);
        let windows = plan_windows(&samples, VadMode::GateAndTrim);

        for w in &windows {
            assert_eq!(
                (w.keep_from, w.keep_until),
                (None, None),
                "{pause} s: {w:?}"
            );
        }
        assert!(
            every_clip_decoded(&windows, &spans),
            "{pause} s: {windows:?}"
        );
        let got = whistle().transcribe(&samples, &options).unwrap();
        let recalls = clip_recalls(&samples, &spans, &options, &got.text);
        let recall = min_recall(&format!("{pause} s pause"), &recalls);
        assert!(
            recall >= MIN_CLIP_RECALL,
            "{pause} s: recall {recall:.2}: {}",
            got.text
        );
    }
}
