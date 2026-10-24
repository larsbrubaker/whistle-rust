// Unit tests for the streaming policy's pure parts (`stream.rs`, `realign.rs`): the word
// normalizer agreement compares with, and the character alignment that finds where the
// committed text ends in a new pass, on every case word-level matching got wrong; plus the
// dropped-audio report at the 30 s cap, from the state that causes it.

use super::realign::{cut, rest_of, says_nothing, Cut};
use super::{normalize, MAX_SAMPLES};

/// The pass's words after the committed text, the first as the rest of its word when the cut
/// is inside it (marked `+`, glued to the committed word before it).
fn new_text(committed: &str, pass: &str) -> Vec<String> {
    let marks = committed
        .split_whitespace()
        .rev()
        .take_while(|w| normalize(w).is_empty())
        .count();
    let last = committed
        .split_whitespace()
        .rev()
        .map(normalize)
        .find(|w| !w.is_empty())
        .unwrap_or_default();
    let committed: String = committed.split_whitespace().map(normalize).collect();
    let words: Vec<&str> = pass.split_whitespace().collect();
    let spelled: Vec<String> = words.iter().map(|w| normalize(w)).collect();
    let Cut { word, offset, .. } = cut(&committed, &last, &spelled, marks);
    let mut out: Vec<String> = words[word.min(words.len())..]
        .iter()
        .map(|w| w.to_string())
        .collect();
    if offset > 0 {
        out[0] = format!("+{}", rest_of(words[word], offset));
    }
    out
}

#[test]
fn agreement_ignores_case_and_punctuation() {
    assert_eq!(normalize("Printing."), normalize("printing"));
    assert_eq!(normalize("«Bonjour,"), "bonjour");
    assert_ne!(normalize("print"), normalize("printing"));
}

#[test]
fn the_rest_of_a_committed_word_read_longer_survives() {
    // "matter" committed; the next pass reads "Matterhackers": "hackers" is new, glued on.
    assert_eq!(
        new_text("Open the matter", "Open the Matterhackers library"),
        ["+hackers", "library"]
    );
    assert_eq!(
        new_text("Open the matter", "Open the Matterhak library"),
        ["+hak", "library"]
    );
}

#[test]
fn a_short_committed_word_does_not_swallow_the_next_words() {
    assert_eq!(
        new_text("make a", "make the cube and round"),
        ["the", "cube", "and", "round"]
    );
}

#[test]
fn a_committed_word_re_read_as_another_is_not_committed_again() {
    assert_eq!(
        new_text("put the box on the", "put the box on a lid"),
        ["lid"]
    );
}

#[test]
fn a_committed_word_that_starts_a_longer_one_continues_it() {
    assert_eq!(new_text("and the", "and they went"), ["+y", "went"]);
}

#[test]
fn a_last_word_read_differently_is_not_committed_twice() {
    assert_eq!(
        new_text(
            "Send a file to Chevan and Creeche",
            "Send a file to Chevan and Krishtov. Open"
        ),
        ["Open"]
    );
}

#[test]
fn punctuation_only_words_are_never_new_twice() {
    assert_eq!(new_text("Bonjour !", "Bonjour ! Merci"), ["Merci"]);
    assert!(new_text("Bonjour !", "Bonjour !").is_empty());
    assert_eq!(new_text("", "« Bonjour"), ["«", "Bonjour"]);
}

#[test]
fn an_uncommitted_mark_after_the_cut_is_new() {
    // "Bonjour" committed without its "!": the pass's "!" is new, not skipped with it.
    assert_eq!(new_text("Bonjour", "Bonjour ! Ça va"), ["!", "Ça", "va"]);
    assert_eq!(new_text("Bonjour !", "Bonjour ! Ça va"), ["Ça", "va"]);
}

#[test]
fn the_lead_in_fragment_of_a_committed_word_is_not_new() {
    // A pass from a floor inside speech hears the end of "bracket" (committed); the committed
    // text it aligns against holds the words that end after the floor, so the fragment is
    // absorbed.
    assert_eq!(new_text("bracket", "racket as an STL"), ["as", "an", "STL"]);
    assert_eq!(
        new_text("of the bracket", "the bracket as an"),
        ["as", "an"]
    );
}

#[test]
fn a_pass_that_hears_almost_nothing_says_nothing() {
    assert!(says_nothing(20, 0));
    assert!(says_nothing(20, 9));
    assert!(!says_nothing(20, 10));
    assert!(!says_nothing(0, 0));
}

#[test]
fn a_last_letter_shared_with_the_next_word_is_no_continuation() {
    // "about" committed; a pass whose alignment ends after the "t" of "the" does not glue "he".
    assert_eq!(
        new_text("ask me of about", "ask me of about the Ultimaker"),
        ["the", "Ultimaker"]
    );
    assert_eq!(
        new_text("ask me of about", "ask me of abou the Ultimaker"),
        ["the", "Ultimaker"]
    );
}

#[test]
fn committed_words_the_pass_does_not_hear_consume_none_of_it() {
    // A settled pass whose committed text lies outside what it heard: consuming a word costs
    // as much as consuming nothing, so every word stays new.
    assert_eq!(
        new_text("edges", "Please export the bracket"),
        ["Please", "export", "the", "bracket"]
    );
    assert_eq!(new_text("the", "a box is"), ["a", "box", "is"]);
    assert_eq!(new_text("about", "the ultimate"), ["the", "ultimate"]);
}

#[test]
fn a_committed_digit_continued_keeps_the_rest() {
    assert_eq!(
        new_text("Make the cube 2", "Make the cube 20 mm wide"),
        ["+0", "mm", "wide"]
    );
}

#[test]
fn repeated_words_are_counted_not_matched_by_time() {
    assert_eq!(new_text("no no no", "no no no no"), ["no"]);
    assert!(new_text("no no no no", "no no no no").is_empty());
}

#[test]
fn the_cut_cost_counts_the_changed_characters() {
    let spelled: Vec<String> = ["hello", "world"].iter().map(|w| normalize(w)).collect();
    assert_eq!(cut("helloworld", "world", &spelled, 0).cost, 0);
    assert_eq!(cut("bonjour", "bonjour", &spelled, 0).cost, 6);
}

#[test]
fn audio_at_the_cap_with_nothing_to_commit_is_reported_dropped() {
    // The state the passes leave when the detector hears speech in 30 s of audio that the
    // decoder reads no words in: a full buffer, nothing committed, nothing pending.
    let mut stream = crate::transcribe::tests::whistle()
        .stream(&crate::transcribe::Options::streaming())
        .expect("opens");
    stream.buffer = vec![0.0; MAX_SAMPLES];
    stream.decoded_end = MAX_SAMPLES as u64;
    let update = stream.push(&[0.0; 16_000]).expect("pushes");
    assert_eq!(update.dropped, 1.0);
    assert_eq!(stream.dropped(), 1.0);
    assert!(stream.buffered() <= MAX_SAMPLES);
    assert!(update.committed.is_empty());
}

#[test]
fn a_30_s_alignment_is_cheap() {
    // About 500 committed characters against 500 of a pass, the most a 30 s buffer holds.
    let words: Vec<String> = (0..100).map(|i| format!("word{}", i % 7)).collect();
    let committed: String = words.concat();
    let started = std::time::Instant::now();
    let c = cut(&committed[..committed.len() - 20], "", &words, 0);
    let elapsed = started.elapsed();
    println!(
        "{} x {} characters in {elapsed:?}",
        committed.len() - 20,
        committed.len()
    );
    assert_eq!(c.cost, 0);
}
