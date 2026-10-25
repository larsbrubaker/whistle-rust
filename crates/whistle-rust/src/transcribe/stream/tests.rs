// Unit tests for the streaming policy's pure parts (`stream.rs`, `realign.rs`): the word
// normalizer agreement compares with, and the character alignment that finds where the
// committed text ends in a new pass, on every case word-level matching got wrong, and the
// re-reads it takes by time; plus the dropped-audio report at the 30 s cap and the pending
// tail's continuation flag, from the states that cause them.

use super::realign::{cut, re_read, rest_of, says_nothing, Cut};
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

/// Words with times, `(word, start, end)`, as committed or read by a pass.
type Timed<'a> = &'a [(&'a str, f32, f32)];

/// As `new_text`, with each word's time: the pass's words after the committed text, less those
/// that re-read committed words the alignment left unmatched (`re_read`), as `stream.rs` takes
/// them at a boundary cut.
fn new_timed(committed: Timed, pass: Timed) -> Vec<String> {
    let text = |words: Timed| -> String {
        let spelled: Vec<&str> = words.iter().map(|w| w.0).collect();
        spelled.join(" ")
    };
    let normalized: String = committed.iter().map(|w| normalize(w.0)).collect();
    let last = committed
        .iter()
        .rev()
        .map(|w| normalize(w.0))
        .find(|w| !w.is_empty())
        .unwrap_or_default();
    let spelled: Vec<String> = pass.iter().map(|w| normalize(w.0)).collect();
    let c = cut(&normalized, &last, &spelled, 0);
    assert_eq!(c.offset, 0, "{} | {}", text(committed), text(pass));
    let lengths: Vec<(usize, f32)> = committed
        .iter()
        .map(|w| (normalize(w.0).chars().count(), w.2))
        .collect();
    let rest = &pass[c.word.min(pass.len())..];
    let times: Vec<(f32, f32)> = rest.iter().map(|w| (w.1, w.2)).collect();
    let skip = re_read(&lengths, c.unmatched, &times);
    rest[skip..].iter().map(|w| w.0.to_string()).collect()
}

#[test]
fn a_committed_word_re_read_longer_is_not_new() {
    // "mm" committed; the next pass reads the same audio as "millimeters": deleting "mm"
    // costs 2, stretching it over "millimeters" 9, so the cut lands before "millimeters".
    let committed = [("cube", 0.48, 0.72), ("20", 0.72, 1.04), ("mm", 1.04, 1.68)];
    let pass = [
        ("cube", 0.48, 0.72),
        ("20", 0.72, 1.04),
        ("millimeters", 1.04, 1.74),
        ("wide", 1.74, 1.9),
    ];
    assert_eq!(new_timed(&committed, &pass), ["wide"]);
    // A number committed as digits and read again as a word.
    let committed = [
        ("the", 0.32, 0.48),
        ("cube", 0.48, 0.72),
        ("20", 0.72, 1.04),
    ];
    let pass = [
        ("the", 0.32, 0.48),
        ("cube", 0.48, 0.72),
        ("twenty", 0.7, 1.06),
        ("millimeters", 1.06, 1.7),
    ];
    assert_eq!(new_timed(&committed, &pass), ["millimeters"]);
}

#[test]
fn a_committed_word_re_read_as_two_is_not_new() {
    // "cube 20" committed as "Q20", then read "queue 20 mm" (en_samantha, measured).
    let committed = [("Make", 0.22, 0.46), ("the", 0.46, 0.7), ("Q20", 0.7, 1.1)];
    let pass = [
        ("Make", 0.22, 0.46),
        ("the", 0.46, 0.7),
        ("queue", 0.7, 0.9),
        ("20", 0.9, 1.1),
        ("mm", 1.1, 1.82),
    ];
    assert_eq!(new_timed(&committed, &pass), ["mm"]);
}

#[test]
fn words_after_an_unmatched_committed_word_stay_new() {
    // The pass did not hear the committed word: its words come after it in time.
    let committed = [("edges", 2.64, 2.96)];
    let pass = [("Please", 3.1, 3.4), ("export", 3.4, 3.8)];
    assert_eq!(new_timed(&committed, &pass), ["Please", "export"]);
    // "a" committed, then "the cube": "the" said after "a" is new; said where "a" was, a re-read.
    let committed = [("make", 0.2, 0.45), ("a", 0.45, 0.6)];
    let late = [("make", 0.2, 0.45), ("the", 0.62, 0.8), ("cube", 0.8, 1.0)];
    assert_eq!(new_timed(&committed, &late), ["the", "cube"]);
    let same = [
        ("make", 0.2, 0.45),
        ("the", 0.45, 0.62),
        ("cube", 0.62, 0.9),
    ];
    assert_eq!(new_timed(&committed, &same), ["cube"]);
}

#[test]
fn a_new_word_in_a_pass_that_matched_nothing_survives() {
    // A settled restart decodes from 0.1 s before the commit point; no character of the
    // committed "edges" matches, so the pass did not hear it, and its short first word "to",
    // whose midpoint falls before the committed end, is new, not a re-read.
    let committed = [("edges", 2.64, 2.96)];
    let pass = [("to", 2.86, 3.0), ("print", 3.0, 3.4)];
    assert_eq!(new_timed(&committed, &pass), ["to", "print"]);
}

#[test]
fn a_new_word_running_well_past_the_committed_end_survives() {
    // The pass leaves out the committed "it" and times the new "cubes" over its audio and on:
    // its midpoint is before the committed end, but it ends 0.2 s past it.
    let committed = [("move", 0.2, 0.45), ("it", 0.45, 0.7)];
    let pass = [("move", 0.2, 0.45), ("cubes", 0.45, 0.9)];
    assert_eq!(new_timed(&committed, &pass), ["cubes"]);
}

#[test]
fn a_committed_mark_after_a_partly_deleted_word_bounds_nothing() {
    // "edges" committed with a "." of its own; the pass reads "edge and": only the "s" is
    // deleted, so no lettered word is wholly unmatched and "and" is new.
    let committed = [("the", 2.4, 2.64), ("edges", 2.64, 2.96), (".", 2.96, 3.3)];
    let pass = [("the", 2.4, 2.64), ("edge", 2.64, 2.96), ("and", 3.0, 3.2)];
    assert_eq!(new_timed(&committed, &pass), ["and"]);
}

#[test]
fn a_fully_matched_cut_takes_nothing_by_time() {
    // Nothing unmatched: a next word timed early (its start before the committed end) is new.
    let committed = [("make", 0.2, 0.45), ("the", 0.45, 0.7)];
    let pass = [("make", 0.2, 0.45), ("the", 0.45, 0.6), ("cube", 0.6, 0.75)];
    assert_eq!(new_timed(&committed, &pass), ["cube"]);
    let spelled: Vec<String> = ["make", "the", "cube"]
        .iter()
        .map(|w| normalize(w))
        .collect();
    assert_eq!(cut("makethe", "the", &spelled, 0).unmatched, 0);
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

#[test]
fn a_pending_tail_that_continues_the_last_committed_word_says_so() {
    // "2" committed, and the latest pass read "20 mm": its pending tail is the rest of "20"
    // and then "mm", so a caller previewing it after the committed text adds no space.
    let word = |text: &str, start: f32, end: f32| crate::transcribe::Word {
        word: text.to_string(),
        start,
        end,
        probability: 1.0,
    };
    let hyp = |text: &str, start: f32, end: f32, glued: bool| super::Hyp {
        repeated: false,
        word: word(text, start, end),
        tokens: Vec::new(),
        glued,
    };
    let mut stream = crate::transcribe::tests::whistle()
        .stream(&crate::transcribe::Options::streaming())
        .expect("opens");
    stream.words = vec![word("cube", 0.48, 0.72), word("2", 0.72, 0.88)];
    stream.text = "cube 2".to_string();
    stream.pending = vec![hyp("0", 0.88, 1.04, true), hyp("mm", 1.04, 1.68, false)];
    let update = stream.push(&[]).expect("pushes");
    assert_eq!(update.pending, "0 mm");
    assert!(update.pending_continues);
    assert!(!update.continues, "nothing committed");
    // A tail that starts a word of its own does not continue.
    stream.pending = vec![hyp("mm", 1.04, 1.68, false)];
    let update = stream.push(&[]).expect("pushes");
    assert_eq!(update.pending, "mm");
    assert!(!update.pending_continues);
    // No tail, nothing to continue.
    stream.pending.clear();
    assert!(!stream.push(&[]).expect("pushes").pending_continues);
}
