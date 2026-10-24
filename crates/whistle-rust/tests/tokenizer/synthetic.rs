// The encode rules on small synthetic vocabularies, where whistle.cact's own vocabulary may
// never exercise them: equal-score pairs merge leftmost first, the longest of overlapping
// user-defined markers wins, and an unknown multi-byte character falls back to its UTF-8 bytes.

use whistle::tokenizer::Tokenizer;

use crate::support::archive::tokenizer_blob;

const NORMAL: u8 = 0;
const UNKNOWN: u8 = 1;
const CONTROL: u8 = 2;
const USER: u8 = 3;
const BYTE: u8 = 4;

fn specials() -> Vec<(f32, u8, Vec<u8>)> {
    vec![
        (0.0, CONTROL, b"<pad>".to_vec()),
        (0.0, CONTROL, b"</s>".to_vec()),
        (0.0, CONTROL, b"<s>".to_vec()),
        (0.0, UNKNOWN, b"<unk>".to_vec()),
    ]
}

fn tokenizer(byte_fallback: bool, pieces: Vec<(f32, u8, Vec<u8>)>) -> Tokenizer {
    let all: Vec<(f32, u8, Vec<u8>)> = specials().into_iter().chain(pieces).collect();
    let refs: Vec<(f32, u8, &[u8])> = all.iter().map(|(s, t, p)| (*s, *t, p.as_slice())).collect();
    Tokenizer::from_blob(&tokenizer_blob(byte_fallback, &refs)).expect("well-formed blob")
}

#[test]
fn equal_score_pairs_merge_leftmost_first() {
    // ids 4 a, 5 b, 6 ab, 7 ba; "ab" and "ba" score the same.
    let tok = tokenizer(
        false,
        vec![
            (-1.0, NORMAL, b"a".to_vec()),
            (-1.0, NORMAL, b"b".to_vec()),
            (-0.5, NORMAL, b"ab".to_vec()),
            (-0.5, NORMAL, b"ba".to_vec()),
        ],
    );
    // "aba": the pairs "ab" (left) and "ba" (right) tie, so "ab" merges: [ab, a].
    assert_eq!(tok.encode("aba"), vec![6, 4]);
    // "bab": "ba" is now the leftmost of the tie: [ba, b].
    assert_eq!(tok.encode("bab"), vec![7, 5]);
    // A higher score still wins over position.
    let tok = tokenizer(
        false,
        vec![
            (-1.0, NORMAL, b"a".to_vec()),
            (-1.0, NORMAL, b"b".to_vec()),
            (-0.5, NORMAL, b"ab".to_vec()),
            (-0.1, NORMAL, b"ba".to_vec()),
        ],
    );
    assert_eq!(tok.encode("aba"), vec![4, 7]);
}

#[test]
fn the_longest_overlapping_marker_wins() {
    // The shorter marker has the lower id, so only the longest-first order picks "<m>x".
    let tok = tokenizer(
        false,
        vec![
            (0.0, USER, b"<m>".to_vec()),
            (0.0, USER, b"<m>x".to_vec()),
            (-1.0, NORMAL, b"x".to_vec()),
            (-1.0, NORMAL, b"y".to_vec()),
        ],
    );
    assert_eq!(tok.encode("<m>x<m>y"), vec![5, 4, 7]);
    assert_eq!(tok.encode("y<m>xx"), vec![7, 5, 6]);
    assert_eq!(tok.decode(&[5, 4, 7]), "<m>x<m>y");
}

#[test]
fn an_unknown_multibyte_character_falls_back_to_its_bytes() {
    let mut pieces: Vec<(f32, u8, Vec<u8>)> = (0..=255u8)
        .map(|b| (0.0, BYTE, format!("<0x{b:02X}>").into_bytes()))
        .collect();
    pieces.push((-1.0, NORMAL, b"a".to_vec()));
    let tok = tokenizer(true, pieces);
    let byte = |b: u8| 4 + u32::from(b);
    // "é" is C3 A9 and "€" is E2 82 AC; neither is a piece.
    assert_eq!(
        tok.encode("aé€"),
        vec![
            260,
            byte(0xC3),
            byte(0xA9),
            byte(0xE2),
            byte(0x82),
            byte(0xAC)
        ]
    );
    assert_eq!(tok.decode(&tok.encode("aé€")), "aé€");
    // Without byte fallback the same character is the unknown piece.
    let tok = tokenizer(false, vec![(-1.0, NORMAL, b"a".to_vec())]);
    assert_eq!(tok.encode("aé"), vec![4, 3]);
}
