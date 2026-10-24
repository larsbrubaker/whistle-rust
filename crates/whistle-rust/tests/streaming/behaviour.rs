// What a caller relies on from a stream: committed words never change and their times are in
// order and inside the audio; silence commits nothing; a stream never holds more than 30 s,
// commits every clip and never commits a word twice (audio dropped at the cap is
// driven in `src/transcribe/stream/tests.rs`, which can set up the state that causes it), with pauses and without (the force rule,
// also fed in step-misaligned chunks); a stream with the detector off stays short; a fired cancel handle stops it at any pass; bad
// options and samples are errors that keep the committed transcript.

use whistle::{Cancel, Error, Options, VadMode, MAX_SAMPLES};

use crate::run::{char_distance, doubled, feed, greedy, joined, CLIPS};
use crate::shared::{clip, clip_recalls, concat, long_english, min_recall, whistle, ENGLISH, RATE};

#[test]
fn committed_words_never_change_and_times_are_in_order() {
    for name in CLIPS {
        let samples = clip(name);
        let fed = feed(&samples, 1.0, &greedy());
        for pair in fed.transcripts.windows(2) {
            let (before, after) = (&pair[0], &pair[1]);
            assert_eq!(
                after.words[..before.words.len()],
                before.words[..],
                "{name}"
            );
            assert!(after.text.starts_with(&before.text), "{name}");
        }
        let duration = samples.len() as f32 / RATE as f32;
        let mut previous_end = 0.0f32;
        for w in &fed.last.words {
            assert!(w.start >= previous_end && w.end >= w.start, "{name}: {w:?}");
            assert!(w.end <= duration + 1e-3, "{name}: {w:?} past {duration} s");
            previous_end = w.end;
        }
        assert!(!fed.last.words.is_empty(), "{name} commits words");
    }
}

#[test]
fn silence_commits_nothing() {
    let silence = clip("silence");
    for samples in [silence, vec![0.0; 5 * RATE]] {
        let fed = feed(&samples, 1.0, &greedy());
        assert!(fed
            .updates
            .iter()
            .all(|u| u.committed.is_empty() && u.pending.is_empty() && u.language.is_none()));
        assert_eq!(fed.last.language, None);
    }
}

/// Doubled words in `streamed` that the offline decode of the same audio does not have.
fn new_doubles(offline: &str, streamed: &str) -> Vec<String> {
    let known = doubled(offline);
    doubled(streamed)
        .into_iter()
        .filter(|d| !known.contains(d))
        .collect()
}

/// A stream of `count` English clips with 0.5-1 s pauses (the windowing tests' input) stays
/// under 30 s, keeps every clip at least as well as the offline windowed decode, commits no
/// word twice, and its committed text is within `max_distance` characters of the offline
/// decode's (measured: 7 on 5 clips, 42 on 12, where the offline decode reads
/// "Mata Hackers", "Creesh Tough" and the stream "matter hackers", "Krishtov").
fn long_stream(count: usize, max_distance: usize) {
    let (samples, spans) = long_english(count);
    let options = greedy();
    let fed = feed(&samples, 1.0, &options);
    assert!(
        fed.max_buffered <= MAX_SAMPLES,
        "held {} samples",
        fed.max_buffered
    );
    let offline = whistle().transcribe(&samples, &options).expect("decodes");
    let windowed = min_recall(
        "offline",
        &clip_recalls(&samples, &spans, &options, &offline.text),
    );
    let recall = min_recall(
        "stream",
        &clip_recalls(&samples, &spans, &options, &fed.last.text),
    );
    println!("offline  {}\nstreamed {}", offline.text, fed.last.text);
    // Measured on 12 clips (45 s): every clip keeps at least 0.71 of its words streamed, 0.57
    // in the offline windowed decode (the stream decodes each utterance from its own start, so
    // it never cuts inside a clip).
    assert!(
        recall >= windowed,
        "a clip kept {recall:.2} of its words, offline {windowed:.2}"
    );
    // "matter" committed, then read as "Matterhak" by the next pass, was committed twice.
    let doubles = new_doubles(&offline.text, &fed.last.text);
    assert!(doubles.is_empty(), "words committed twice: {doubles:?}");
    assert_eq!(fed.dropped, 0.0);
    let distance = char_distance(&offline.text, &fed.last.text);
    println!("character distance to the offline decode {distance}");
    assert!(
        distance <= max_distance,
        "{distance} characters from the offline decode"
    );
}

#[test]
fn a_long_stream_commits_every_clip_and_never_twice() {
    // 5 clips, 18 s: holds the "matter" case at 11.5 s.
    long_stream(5, 7);
}

#[test]
#[ignore = "45 s stream, about a minute; run with --ignored"]
fn a_45_s_stream_stays_under_30_s_commits_every_clip_and_never_twice() {
    long_stream(12, 42);
}

/// `count` English clips with no pause the endpoint rule could use, streamed at `step`: only
/// the force rule commits. The worst clip keeps at least `min_recall` of its words, no word is
/// committed twice, the committed text is within `max_distance` characters of the offline
/// decode's, and 0.37 s chunks (which do not divide the
/// step) give the same text.
fn no_pause(count: usize, step: f32, bound: f64, max_distance: usize) {
    let clips: Vec<(&str, f32)> = (0..count).map(|i| (ENGLISH[i % 5], 0.0)).collect();
    let (samples, spans) = concat(&clips);
    let options = Options {
        stream_step: step,
        ..greedy()
    };
    let fed = feed(&samples, 1.0, &options);
    assert!(
        fed.max_buffered <= MAX_SAMPLES,
        "held {} samples",
        fed.max_buffered
    );
    let recall = min_recall(
        "no pause",
        &clip_recalls(&samples, &spans, &options, &fed.last.text),
    );
    println!("streamed {}", fed.last.text);
    assert!(recall >= bound, "a clip kept {recall:.2} of its words");
    // A forced cut started the next pass inside a committed word: "of" was committed twice.
    let doubles = doubled(&fed.last.text);
    assert!(doubles.is_empty(), "words committed twice: {doubles:?}");
    // Passes run at step boundaries counted from the stream's start, and room is made for a
    // whole step at once, so a chunk size that does not divide the step changes nothing.
    let offline = whistle().transcribe(&samples, &options).expect("decodes");
    let distance = char_distance(&offline.text, &fed.last.text);
    println!(
        "offline  {}\ncharacter distance to the offline decode {distance}",
        offline.text
    );
    assert!(
        distance <= max_distance,
        "{distance} characters from the offline decode"
    );
    let misaligned = feed(&samples, 0.37, &options);
    assert_eq!(joined(&misaligned), joined(&fed));
    assert!(misaligned.max_buffered <= MAX_SAMPLES);
}

#[test]
fn speech_with_no_pause_is_force_committed_once_in_any_chunking() {
    // 33 s in 10 s steps: the force rule at a third of the cost of 3 s steps (passes at 10, 20
    // and 30 s, a forced cut, the rest at finish). Measured: 4 of 7 words for the worst clip,
    // 25 characters from the offline decode (3 of 7 and 28 while a pass that heard none of the
    // committed text consumed its first word).
    no_pause(11, 10.0, 4.0 / 7.0, 25);
}

#[test]
#[ignore = "33 s twice at 1 s steps, about a minute; run with --ignored"]
fn speech_with_no_pause_at_1_s_steps_is_force_committed_once() {
    // Measured 0.57 for the worst clip; 49 characters from the offline decode (42 before the
    // deterministic math moved the decode to 43 on the merged code, 49 with the lead-in words in
    // the aligned text and the stricter mid-word rule).
    no_pause(11, 1.0, 0.57, 49);
}

/// A stream of `count` English clips with the detector off: without pauses to start at, each
/// pass starts 1 s before the commit point, so the buffer stays short and commits keep flowing
/// (a restart repeated on every pass once stopped them for 22 s and let the buffer climb to
/// 29.5 s); the committed text is within `max_distance` characters of the offline decode's.
fn off_stream(count: usize, max_held: f32, max_gap: f32, max_distance: usize) {
    let (samples, _) = long_english(count);
    let options = Options {
        vad: VadMode::Off,
        ..greedy()
    };
    let fed = feed(&samples, 1.0, &options);
    let offline = whistle().transcribe(&samples, &options).expect("decodes");
    let held = fed.max_buffered as f32 / RATE as f32;
    // The longest stretch of audio received without a commit.
    let mut gap = 0.0f32;
    let mut last = 0.0f32;
    for (u, &received) in fed.updates.iter().zip(&fed.received) {
        if !u.committed.is_empty() {
            gap = gap.max(received - last);
            last = received;
        }
    }
    let distance = char_distance(&offline.text, &fed.last.text);
    println!("offline  {}\nstreamed {}", offline.text, fed.last.text);
    println!("held at most {held:.2} s, longest gap {gap:.2} s, character distance {distance}");
    assert!(held <= max_held, "held {held:.2} s");
    assert!(gap <= max_gap, "{gap:.2} s without a commit");
    assert!(
        distance <= max_distance,
        "{distance} characters from the offline decode"
    );
}

#[test]
fn a_stream_with_the_detector_off_stays_short_and_keeps_committing() {
    // 5 clips, 18 s. Measured: at most 4.74 s held, 2.0 s at most between commits, 18
    // characters from the offline decode (each pass starts inside a word; 25 before a pass
    // that hears none of the committed text stopped consuming its first word).
    off_stream(5, 5.0, 2.0, 18);
}

#[test]
#[ignore = "45 s with the detector off; run with --ignored"]
fn a_45_s_stream_with_the_detector_off_stays_short_and_keeps_committing() {
    // Measured: at most 9.2 s held, 5.0 s at most between commits, 64 characters from the
    // offline windowed decode (7.52 s, 6.0 s and 61 while a pass that heard none of the
    // committed text consumed its first word: keeping that word new costs agreement a pass).
    off_stream(12, 9.25, 5.0, 64);
}

#[test]
fn cancel_mid_stream_is_cancelled() {
    let cancel = Cancel::new();
    let options = Options {
        cancel: Some(cancel.clone()),
        ..greedy()
    };
    let samples = clip("en_daniel");
    let mut stream = whistle().stream(&options).expect("opens");
    stream.push(&samples[..RATE]).expect("first second decodes");
    cancel.cancel();
    assert!(matches!(
        stream.push(&samples[RATE..2 * RATE]),
        Err(Error::Cancelled)
    ));
    assert!(matches!(stream.finish(), Err(Error::Cancelled)));
}

#[test]
fn errors_keep_the_committed_transcript() {
    let samples = clip("en_daniel");
    let mut stream = whistle().stream(&greedy()).expect("opens");
    stream
        .push(&samples[..samples.len() - RATE / 2])
        .expect("decodes");
    let before = stream.transcript();
    assert!(!before.words.is_empty(), "agreement committed words");
    assert!(matches!(
        stream.push(&[0.0, f32::NAN]),
        Err(Error::Input(_))
    ));
    assert_eq!(stream.transcript(), before);
}

#[test]
fn bad_stream_options_are_input_errors() {
    for step in [0.0, 0.05, -1.0, f32::NAN, 11.0] {
        let options = Options {
            stream_step: step,
            ..greedy()
        };
        assert!(
            matches!(whistle().stream(&options), Err(Error::Input(_))),
            "{step}"
        );
    }
    let options = Options {
        language: Some("xx".into()),
        ..greedy()
    };
    assert!(matches!(whistle().stream(&options), Err(Error::Input(_))));
}
