// Streams over garbled audio (docs/design/streaming.md "A pass that loops"): de_anna with
// 128 ms blocks replayed as a late audio callback replays them (`support::garble`). Its last
// pass, over the whole clip, ran into a loop ("B" and "ib" 317 times) and, as the stream's
// final pass, committed all of it. With the repetition guard the pass is rolled back past the
// loop and reads on, and its words commit as any pass's do. A garbled start followed by
// clean speech, with no pause, makes the passes start in the loop: the speech after it still
// commits as it arrives.

use std::collections::HashMap;

use whistle::Options;

use crate::run::{feed, greedy, Fed};
use crate::shared::clip;
use crate::support::garble::{longest_repetition, replayed_blocks};
use crate::wer::normalize;

fn garbled_de_anna() -> Vec<f32> {
    replayed_blocks(&clip("de_anna"), 2, 2048)
}

fn three_beams() -> Options {
    Options {
        beams: 3,
        ..Options::streaming()
    }
}

/// The most consecutive copies of any block of committed words.
fn most_copies(fed: &Fed) -> usize {
    let mut ids = HashMap::new();
    let words: Vec<u32> = normalize(&fed.last.text)
        .into_iter()
        .map(|w| {
            let next = ids.len() as u32;
            *ids.entry(w).or_insert(next)
        })
        .collect();
    longest_repetition(&words).0
}

#[test]
fn a_pass_that_loops_is_rolled_back_and_its_words_commit() {
    let samples = garbled_de_anna();
    let cases = [
        (
            greedy(),
            "Biby Biby macht den Würfetz zweit Senti Mimi eta Höhe.",
        ),
        (
            three_beams(),
            "Bibby Bibby macht den Würfetz zwei Zentimeter höher.",
        ),
    ];
    for (options, want) in cases {
        let fed = feed(&samples, 1.0, &options);
        assert!(
            most_copies(&fed) < 5,
            "{} beams: {:?}",
            options.beams,
            fed.last.text
        );
        assert_eq!(fed.last.text, want, "{} beams", options.beams);
    }
}

#[test]
fn speech_after_a_garbled_start_commits_while_the_passes_loop_from_the_start() {
    let mut samples = garbled_de_anna();
    let speech = clip("de_anna");
    samples.extend_from_slice(&speech);
    samples.extend_from_slice(&speech);
    // One beam, the live preset; three beams commit the same way (measured), at three times
    // the cost.
    let fed = feed(&samples, 1.0, &greedy());
    // Words commit before the end, from passes the guard rolled back.
    let early: usize = fed.updates[..fed.updates.len() - 1]
        .iter()
        .map(|u| u.committed.len())
        .sum();
    assert!(early >= 10, "{early} words before finish");
    assert!(fed.transcripts.iter().any(|t| t.repeated));
    // Both clean readings of the sentence are there, and no loop.
    let words = normalize(&fed.last.text);
    let bitte = words.iter().filter(|w| *w == "bitte").count();
    assert!(bitte >= 2, "{:?}", fed.last.text);
    assert!(words.len() >= 18, "{:?}", fed.last.text);
    assert!(most_copies(&fed) < 5, "{:?}", fed.last.text);
}
