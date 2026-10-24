// Unit tests for the keyword trie and matcher (`keywords.rs`): `deltas` agrees with `advance`
// for every id, the adjustments a path collects always equal the matcher's kept plus partial
// bonus, keywords that are prefixes of each other, blank entries, and spellings that tokenize
// alike sharing one path, and byte pieces classified by their byte. The tokenizer tests read
// models/whistle.cact.

use std::path::Path;

use super::{spellings, KeywordTrie, Matcher};
use crate::cact::Archive;
use crate::tokenizer::{PieceType, Tokenizer};

const EOS: u32 = 1;
const B: f64 = 3.0;

/// Tokens: 2 `▁x`, 3 `▁b`, 4 `c`, 5 `▁a`, 6 `d`, 7 `e`, 8 `-`.
fn edges() -> Vec<(Option<char>, Option<char>)> {
    let p = |f: char, l: char| (Some(f), Some(l));
    vec![
        (None, None),
        (None, None),
        p(' ', 'x'),
        p(' ', 'b'),
        p('c', 'c'),
        p(' ', 'a'),
        p('d', 'd'),
        p('e', 'e'),
        p('-', '-'),
    ]
}

fn trie() -> KeywordTrie {
    // `▁a d`, `▁a d e` (a prefix of it), `▁b c`, and `c` unspaced (after a hyphen).
    KeywordTrie::build(
        &[vec![5, 6], vec![5, 6, 7], vec![3, 4], vec![4]],
        &edges(),
        B as f32,
    )
}

/// Walks `path`, checking at every step that `deltas` matches `advance` for every id and that
/// the collected adjustments equal kept + partial. Returns the final matcher.
fn walk(t: &KeywordTrie, path: &[u32]) -> Matcher {
    let mut m = Matcher::start();
    let mut collected = 0.0;
    for &token in path {
        let d = t.deltas(&m, 9, EOS);
        for id in 0..9 {
            assert_eq!(
                d[id as usize],
                t.advance(&m, id, EOS).1,
                "id {id} after {path:?}"
            );
        }
        let (next, delta, _) = t.advance(&m, token, EOS);
        collected += delta;
        m = next;
        assert!(
            (collected - (m.committed() + m.partial())).abs() < 1e-12,
            "{path:?}"
        );
    }
    m
}

#[test]
fn deltas_match_advance_and_adjustments_never_leak() {
    let t = trie();
    for path in [
        vec![5, 6, 7, 2],
        vec![5, 6, 2],
        vec![5, 6, 4],
        vec![5, 3, 4, EOS],
        vec![2, 8, 4, 2],
        vec![4, 5, 2, 5, 6, EOS],
        vec![3, 5, 6, 7, 6, 2],
    ] {
        walk(&t, &path);
    }
}

#[test]
fn keywords_that_are_prefixes_of_each_other_keep_the_longest_completed_one() {
    let t = trie();
    assert_eq!(walk(&t, &[5, 6, 7, 2]).committed(), 3.0 * B);
    assert_eq!(walk(&t, &[5, 6, 2]).committed(), 2.0 * B);
    // `▁a d` continued by the letter `c`: a different word, nothing kept.
    let m = walk(&t, &[5, 6, 4, 2]);
    assert_eq!((m.committed(), m.partial()), (0.0, 0.0));
}

#[test]
fn abandoning_reports_retraction_only_for_partial_keywords() {
    let t = trie();
    let m = walk(&t, &[5]);
    assert!(t.advance(&m, 2, EOS).2, "▁a then ▁x abandons");
    assert!(t.advance(&m, 3, EOS).2, "▁a then ▁b abandons A for B");
    let done = walk(&t, &[5, 6]);
    assert!(!t.advance(&done, 2, EOS).2, "a completed keyword is kept");
    assert!(!t.advance(&done, EOS, EOS).2);
}

fn tokenizer() -> Tokenizer {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/whistle.cact");
    let bytes = std::fs::read(root).expect("models/whistle.cact");
    Tokenizer::from_archive(&Archive::parse(&bytes).unwrap()).unwrap()
}

#[test]
fn blank_and_space_only_keywords_build_no_trie() {
    let t = tokenizer();
    let blank = vec![String::new(), "   ".to_string(), "\t".to_string()];
    assert!(KeywordTrie::new(&t, &blank, 8192, 3.0).is_none());
    assert!(KeywordTrie::new(&t, &[], 8192, 3.0).is_none());
}

#[test]
fn spellings_that_tokenize_alike_share_one_path() {
    assert_eq!(spellings("PETG"), ["PETG", "petg"]);
    assert_eq!(spellings("cura"), ["Cura", "cura"]);
    let t = tokenizer();
    let one = KeywordTrie::new(&t, &["Cura".to_string()], 8192, 3.0).unwrap();
    let both = KeywordTrie::new(
        &t,
        &["Cura".to_string(), "cura".to_string(), " Cura ".to_string()],
        8192,
        3.0,
    )
    .unwrap();
    assert_eq!(one.len(), both.len());
}

#[test]
fn byte_pieces_are_boundaries_for_ascii_space_and_punctuation_and_letters_otherwise() {
    let t = tokenizer();
    let trie = KeywordTrie::new(&t, &["x".to_string()], 8192, 3.0).unwrap();
    let byte_id =
        |want: u8| (0..8192u32).find(|&id| t.piece_type(id) == Some(PieceType::Byte(want)));
    let mut seen = 0;
    for b in *b" -,.!" {
        if let Some(id) = byte_id(b) {
            seen += 1;
            assert!(trie.boundary[id as usize], "byte {b:#x} starts a word");
            assert!(
                trie.ends_open[id as usize],
                "byte {b:#x} lets a keyword follow"
            );
        }
    }
    for b in [b'a', b'Z', 0xc3, 0x85, 0xff] {
        if let Some(id) = byte_id(b) {
            seen += 1;
            assert!(!trie.boundary[id as usize], "byte {b:#x} continues a word");
            assert!(
                !trie.ends_open[id as usize],
                "byte {b:#x} ends inside a word"
            );
        }
    }
    assert!(
        seen >= 4,
        "the tokenizer has byte pieces for these bytes ({seen} found)"
    );
}
