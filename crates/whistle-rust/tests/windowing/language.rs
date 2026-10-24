// The language over windows: the first window that keeps at least 3 words detects it and the
// rest are decoded with it forced. An English part followed by a German one (about 24 s and
// 6 s, cut at the 2 s pause between them) comes out labelled English, its German window decoded
// exactly as that window alone with English forced (alone it is German). A German fragment of
// under 3 words, then a long pause and English, does not fix the language: the English window
// is decoded with detection, and the transcript is English. (A long English input staying
// English is pinned in `pauses`.)

use whistle::{plan_windows, Options, VadMode};

use crate::shared::{alone, clip, concat, greedy, whistle, ENGLISH, RATE};

/// About 24 s of English, a 2 s pause, then about 6 s of German.
fn english_then_german() -> Vec<f32> {
    let mut clips: Vec<(&str, f32)> = (0..7).map(|i| (ENGLISH[i % 5], 0.5)).collect();
    clips.last_mut().unwrap().1 = 2.0;
    clips.extend((0..2).map(|_| ("de_anna", 0.5)));
    concat(&clips).0
}

#[test]
fn the_first_windows_language_is_forced_on_the_rest() {
    let samples = english_then_german();
    let options = greedy(VadMode::GateAndTrim, false);
    let windows = plan_windows(&samples, VadMode::GateAndTrim);
    assert_eq!(windows.len(), 2, "{windows:?}");
    let got = whistle().transcribe(&samples, &options).unwrap();
    assert_eq!(got.language.as_deref(), Some("en"));
    let first = alone(&samples, &windows[0], &options);
    assert_eq!(first.language.as_deref(), Some("en"));
    let forced = Options {
        language: Some("en".into()),
        ..options.clone()
    };
    let second = alone(&samples, &windows[1], &forced);
    // On its own the German window is German; inside the long input it is decoded as English.
    let detected = alone(&samples, &windows[1], &options);
    assert_eq!(detected.language.as_deref(), Some("de"));
    assert_eq!(
        got.text,
        format!("{} {}", first.text.trim(), second.text.trim())
    );
}

#[test]
fn a_fragment_does_not_fix_the_language() {
    // The first 0.5 s of de_anna, a 30 s pause, then three English clips.
    let mut samples = clip("de_anna");
    samples.truncate(RATE / 2);
    samples.resize(samples.len() + 30 * RATE, 0.0);
    let (english, _) = concat(&[(ENGLISH[0], 0.5), (ENGLISH[1], 0.5), (ENGLISH[2], 0.0)]);
    samples.extend(english);
    let options = greedy(VadMode::GateAndTrim, false);
    let windows = plan_windows(&samples, VadMode::GateAndTrim);
    assert_eq!(windows.len(), 2, "{windows:?}");
    let fragment = alone(&samples, &windows[0], &options);
    let words = fragment.text.split_whitespace().count();
    println!("fragment {:?} ({:?})", fragment.text, fragment.language);
    assert!((1..3).contains(&words), "{:?}", fragment.text);
    let got = whistle().transcribe(&samples, &options).unwrap();
    assert_eq!(got.language.as_deref(), Some("en"), "{}", got.text);
    let rest = alone(&samples, &windows[1], &options);
    assert_eq!(
        got.text,
        format!("{} {}", fragment.text.trim(), rest.text.trim())
    );
}
