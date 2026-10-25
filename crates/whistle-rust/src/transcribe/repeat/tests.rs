// Unit tests for the repetition guard (`repeat.rs`): the rule on synthetic token sequences
// with no word starts (where a loop starts, how many copies it needs at each block length, the
// copy kept), the guard's roll-backs, bans and final cut, and the rule on real text through
// the archive's tokenizer, checked after every token as the search checks it: the runaway
// French loop fires and keeps its first copy, a loop whose first copy is capitalised is cut
// at a word boundary, and repetition people really say, in all seven languages (doubled
// words, chants, counting, a phrase said four times), never fires. Uses models/whistle.cact
// from the workspace root.

use super::{looped, Guard, Pieces, Verdict, MAX_REWINDS, MIN_COPIES, MIN_SPAN, MIN_WORDS};
use crate::transcribe::tests::whistle;

/// Where the guard first fires on `tokens` fed one at a time, as the search feeds them: the
/// prefix length it fired at and the length it keeps.
fn first_fire(tokens: &[u32], pieces: &Pieces) -> Option<(usize, usize)> {
    (1..=tokens.len()).find_map(|end| looped(&tokens[..end], pieces).map(|keep| (end, keep)))
}

#[test]
fn a_one_token_loop_inside_a_word_fires_at_its_thirtieth_copy_and_keeps_the_first() {
    // 7 then 3 repeated: "B" then "ib" over and over.
    let mut tokens = vec![7];
    tokens.extend([3; 40]);
    assert_eq!(first_fire(&tokens, &Pieces::NONE), Some((1 + MIN_SPAN, 2)));
    assert_eq!(
        looped(&tokens[..MIN_SPAN], &Pieces::NONE),
        None,
        "29 copies"
    );
}

#[test]
fn a_loop_spans_five_blocks_and_thirty_tokens() {
    for n in 1..=12usize {
        let block: Vec<u32> = (10..10 + n as u32).collect();
        let tokens: Vec<u32> = std::iter::repeat_n(block, 40).flatten().collect();
        let span = (MIN_COPIES * n).max(MIN_SPAN);
        assert_eq!(
            first_fire(&tokens, &Pieces::NONE),
            Some((span, n)),
            "a {n}-token block fires at {span} tokens"
        );
    }
}

#[test]
fn a_loop_entered_part_way_into_its_block_keeps_one_whole_copy_from_its_start() {
    // 1 2, then the loop 5 6 7 5 6 7 …: the tail's blocks are cut at the last token, but the
    // copy kept starts where the repetition starts.
    let mut tokens = vec![1, 2];
    for _ in 0..10 {
        tokens.extend([5, 6, 7]);
    }
    tokens.push(5);
    assert_eq!(looped(&tokens, &Pieces::NONE), Some(5));
}

#[test]
fn a_repetition_that_ended_is_not_a_loop() {
    let mut tokens = vec![3; 29];
    tokens.extend([4, 5, 6]);
    assert_eq!(first_fire(&tokens, &Pieces::NONE), None);
    assert_eq!(looped(&[], &Pieces::NONE), None);
}

/// Ids 0..10 carry text, 2 to 5 start a word (`▁…`), 6 is a comma.
fn pieces() -> Pieces {
    let mut firsts = vec![Some('a'); 10];
    for f in &mut firsts[2..6] {
        *f = Some(' ');
    }
    firsts[6] = Some(',');
    Pieces::from_firsts(&firsts)
}

#[test]
fn a_one_word_block_needs_twenty_copies() {
    // "▁ja" (2) and "," (6): two tokens, one word per copy.
    let tokens: Vec<u32> = std::iter::repeat_n([2, 6], 30).flatten().collect();
    let (at, keep) = first_fire(&tokens, &pieces()).expect("fires");
    // The 30-token span holds 15 words; the 20th word start is token 39.
    assert_eq!((at, keep), (2 * (MIN_WORDS - 1) + 1, 2));
}

#[test]
fn a_cut_inside_a_word_moves_back_to_the_word_start_in_the_copy() {
    // "W" (7) "ięcej" (8) "," (6), then "▁w" (2) "ięcej" "," for ever: the repetition starts at
    // "ięcej", so its first copy "ięcej , ▁w" ends in the head of a word.
    let mut tokens = vec![7];
    for _ in 0..30 {
        tokens.extend([8, 6, 2]);
    }
    tokens.push(8);
    // Cut before the "▁w", so the kept text is "Więcej,".
    assert_eq!(first_fire(&tokens, &pieces()).map(|f| f.1), Some(3));
}

#[test]
fn the_guard_rolls_back_and_bans_three_times_then_cuts() {
    let mut tokens = vec![7];
    tokens.extend([3; 29]);
    let mut guard = Guard::default();
    assert_eq!(guard.check(&tokens, &Pieces::NONE), None);
    assert!(!guard.fired());
    tokens.push(3);
    for _ in 0..MAX_REWINDS {
        assert_eq!(
            guard.check(&tokens, &Pieces::NONE),
            Some(Verdict::Rewind(2))
        );
        assert!(guard.fired());
    }
    // The token that started the second copy is banned where it stood, once per roll-back.
    assert_eq!(guard.banned(2), vec![3; MAX_REWINDS]);
    assert!(guard.banned(1).is_empty());
    assert_eq!(guard.check(&tokens, &Pieces::NONE), Some(Verdict::Cut(2)));
}

#[test]
fn a_roll_back_to_an_earlier_point_drops_the_bans_after_it() {
    let mut guard = Guard::default();
    let mut tokens = vec![7, 8];
    tokens.extend([3; 30]);
    assert_eq!(
        guard.check(&tokens, &Pieces::NONE),
        Some(Verdict::Rewind(3))
    );
    let mut earlier = vec![7];
    earlier.extend([9; 30]);
    assert_eq!(
        guard.check(&earlier, &Pieces::NONE),
        Some(Verdict::Rewind(2))
    );
    assert!(guard.banned(3).is_empty());
    assert_eq!(guard.banned(2), [9]);
}

fn real() -> Pieces {
    let w = whistle();
    Pieces::new(w.tokenizer(), w.first_tag)
}

fn tokens(text: &str) -> Vec<u32> {
    whistle().tokenizer().encode(text)
}

fn decode(tokens: &[u32]) -> String {
    whistle().tokenizer().decode(tokens)
}

#[test]
fn the_runaway_french_loop_fires_and_keeps_its_first_copy() {
    let mut text = String::from("et les trucs, je trouve que je trouve le même mémoire,");
    for _ in 0..59 {
        text.push_str(" je trouve que");
    }
    let tokens = tokens(&text);
    let (at, keep) = first_fire(&tokens, &real()).expect("the loop fires");
    assert_eq!(
        decode(&tokens[..keep]),
        "et les trucs, je trouve que je trouve le même mémoire, je trouve que"
    );
    // Measured: after 166 characters (the seventh copy's second word, twenty words of loop);
    // the 873-character decode's zlib compression ratio first passes 2.4 (Whisper's rejection
    // threshold) at about 130.
    let fired = decode(&tokens[..at]);
    assert!(fired.chars().count() < 180, "fired at {at}: {fired:?}");
}

#[test]
fn a_loop_whose_first_copy_is_capitalised_keeps_whole_words() {
    for (word, kept) in [
        ("Więcej", "Więcej,"),
        ("Dość", "Dość,"),
        ("Sì", "Sì,"),
        ("Più", "Più,"),
    ] {
        let mut text = format!("{word},");
        for _ in 0..40 {
            text.push_str(&format!(" {},", word.to_lowercase()));
        }
        let tokens = tokens(&text);
        let (_, keep) = first_fire(&tokens, &real()).unwrap_or_else(|| panic!("{word}"));
        assert_eq!(decode(&tokens[..keep]), kept, "{word} ({tokens:?})");
    }
}

#[test]
fn legitimate_repetition_never_fires() {
    let phrases = [
        "No, no, no.",
        "no no no no no",
        "one two one two",
        "One two, one two, one two, one two.",
        "one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve",
        "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20",
        "I'm sorry, I'm sorry, I'm sorry, I'm sorry.",
        "Testing one two three, testing one two three, testing one two three.",
        "ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha",
        "la la la la la la la la la la la la la la la la",
        "very very very very very good",
        "Undo, undo, undo, undo, undo, undo, undo, undo.",
        "Zoom in. Zoom in. Zoom in. Zoom in.",
        "Make it 1000 000 000 000 mm wide.",
        "Move it left, left, left, left, then up, up, up, up.",
        // German
        "Ja, ja, ja, ja, ja, ja, ja, ja, ja, ja, ja, ja, ja, ja, ja, ja.",
        "Bitte bitte bitte, nein nein nein nein.",
        // French
        "Non, non, non, non, non !",
        "Hé, hé, hé, hé, hé, hé, hé, hé.",
        "Ça, ça, ça, ça, ça, ça.",
        // Spanish
        "Sí, sí, sí, sí, sí, sí, sí, sí, sí, sí.",
        "Más, más, más, más, más, más.",
        // Italian
        "Sì, sì, sì, sì, sì, sì, sì, sì, sì, sì, sì, sì.",
        "Più, più, più, più, più, più, più.",
        "Più su, più su, più su, più su, più su.",
        "Così, così, così, così, così, così.",
        "Ancora, ancora, ancora, ancora, ancora, ancora.",
        // Dutch
        "Nee nee nee nee nee nee nee nee nee nee.",
        "Ja hoor, ja hoor, ja hoor, ja hoor.",
        "Hoger, hoger, hoger, hoger, hoger, hoger.",
        // Polish
        "Więcej, więcej, więcej, więcej, więcej, więcej.",
        "Dość, dość, dość, dość, dość, dość.",
        "Już, już, już, już, już, już.",
        "Tak, tak, tak, tak, tak, tak, tak, tak, tak, tak.",
        "Nie, nie, nie, nie, nie, nie.",
        "Zwiększ, zwiększ, zwiększ, zwiększ, zwiększ.",
        "Przesuń w lewo, przesuń w lewo, przesuń w lewo, przesuń w lewo.",
    ];
    let pieces = real();
    for phrase in phrases {
        let tokens = tokens(phrase);
        assert_eq!(
            first_fire(&tokens, &pieces),
            None,
            "{phrase:?} ({tokens:?})"
        );
    }
}
