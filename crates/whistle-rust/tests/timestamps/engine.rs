// The words and their times against the engine's. On the 8 contract clips at the default 3
// beams (one test, so one decode per clip): the text is the engine's, the words are the
// engine's word list, every start and end is within one 80 ms frame of the engine's, and the
// times are well formed (ordered, contiguous, inside the audio). On the held-out extra clips
// (not used to choose the heads), wherever the text is the engine's: within one frame except at
// most one time per clip within two.
//
// Measured 2026-10-24 on the contract clips: of the 146 starts and ends, 111 equal the engine's
// frame, 22 are one frame early and 13 one frame late. Most of the early ones are the first
// word: the engine starts it at 0.08 s, the port at the start of the decoded audio.

use whistle::{Transcript, FRAME_SECONDS};

use crate::shared::{engine, extra_clips, speech_clips, timed, wav, wav_in, whistle};

/// Float slack on top of the frame bounds: both sides are multiples of 0.08 s in f32, ours
/// plus the trim offset.
const EPS: f32 = 1e-4;

/// Each time's distance from the engine's in frames, starts and ends, after checking the word
/// list matches.
fn frame_errors(clip: &str, t: &Transcript, words: &[crate::shared::EngineWord]) -> Vec<f32> {
    let ours: Vec<&str> = t.words.iter().map(|w| w.word.as_str()).collect();
    let theirs: Vec<&str> = words.iter().map(|w| w.word.as_str()).collect();
    assert_eq!(ours, theirs, "{clip}: the word list");
    t.words
        .iter()
        .zip(words)
        .flat_map(|(w, e)| [(w.start - e.start).abs(), (w.end - e.end).abs()])
        .map(|d| (d - EPS).max(0.0) / FRAME_SECONDS)
        .collect()
}

fn check_well_formed(clip: &str, t: &Transcript, duration: f32) {
    assert!(!t.words.is_empty(), "{clip}: no words");
    for w in &t.words {
        assert!(
            w.start >= 0.0 && w.end >= w.start && w.end <= duration + EPS,
            "{clip}: {w:?} outside 0..{duration} s"
        );
        assert!(
            w.probability > 0.0 && w.probability <= 1.0,
            "{clip}: {w:?} probability"
        );
    }
    for pair in t.words.windows(2) {
        assert!(
            pair[0].start <= pair[1].start,
            "{clip}: starts out of order {pair:?}"
        );
        assert_eq!(
            pair[0].end, pair[1].start,
            "{clip}: not contiguous {pair:?}"
        );
    }
}

#[test]
fn contract_clips_words_and_times_match_the_engine_within_a_frame() {
    let mut worst = (0.0f32, String::new());
    for clip in speech_clips() {
        let bytes = wav(&clip);
        let t = whistle()
            .transcribe_wav(&bytes, &timed())
            .expect("transcribes");
        let (text, words) = engine("engine", &clip);
        assert_eq!(t.text, text, "{clip}: the text is the engine's");
        assert_eq!(
            whistle().tokenizer().decode(&t.tokens),
            t.text,
            "{clip}: tokens"
        );
        for (i, e) in frame_errors(&clip, &t, &words).into_iter().enumerate() {
            if e > worst.0 {
                worst = (e, format!("{clip} time {i}"));
            }
            assert!(
                e <= 1.0,
                "{clip}: time {i} is {e:.2} frames from the engine's"
            );
        }
        let audio = whistle::audio::read_wav(&bytes).expect("reads");
        check_well_formed(
            &clip,
            &t,
            audio.samples.len() as f32 / audio.sample_rate as f32,
        );
    }
    println!("worst: {:.2} frames at {}", worst.0, worst.1);
}

#[test]
fn held_out_clips_times_match_the_engine_within_a_frame_or_one_at_two() {
    let mut skipped = Vec::new();
    let mut checked = 0;
    for clip in extra_clips() {
        let bytes = wav_in("clips-extra", &clip);
        let t = whistle()
            .transcribe_wav(&bytes, &timed())
            .expect("transcribes");
        let (text, words) = engine("engine-extra", &clip);
        if t.text != text {
            skipped.push(clip);
            continue;
        }
        checked += 1;
        let errors = frame_errors(&clip, &t, &words);
        let over_one = errors.iter().filter(|&&e| e > 1.0).count();
        assert!(
            over_one <= 1 && errors.iter().all(|&e| e <= 2.0),
            "{clip}: frame errors {errors:?}"
        );
        let audio = whistle::audio::read_wav(&bytes).expect("reads");
        check_well_formed(
            &clip,
            &t,
            audio.samples.len() as f32 / audio.sample_rate as f32,
        );
    }
    println!("checked {checked} held-out clips; text differs from the engine's on {skipped:?}");
    assert!(
        checked >= 3,
        "only {checked} held-out clips match the engine's text: {skipped:?}"
    );
}
