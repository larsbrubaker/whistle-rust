// A malformed tokenizer blob is an error, never a panic: every truncation of a small valid blob,
// an unknown piece type, a byte piece that is not `<0xXX>`, invalid UTF-8, and byte fallback
// without all 256 byte pieces.

use whistle::tokenizer::Tokenizer;
use whistle::Error;

use crate::support::archive::tokenizer_blob as blob;

fn small() -> Vec<u8> {
    blob(
        false,
        &[
            (0.0, 2, b"<pad>"),
            (0.0, 2, b"</s>"),
            (0.0, 2, b"<s>"),
            (0.0, 1, b"<unk>"),
            (-1.0, 0, b"a"),
            (-2.0, 0, b"b"),
            (-0.5, 0, b"ab"),
            (0.0, 3, b"<|en|>"),
        ],
    )
}

fn assert_tokenizer_error(bytes: &[u8], what: &str) {
    match Tokenizer::from_blob(bytes) {
        Err(Error::Tokenizer(message)) => assert!(!message.is_empty(), "{what}"),
        Err(other) => panic!("{what}: unexpected error {other}"),
        Ok(_) => panic!("{what}: parsed"),
    }
}

#[test]
fn the_small_blob_encodes_and_decodes() {
    let tok = Tokenizer::from_blob(&small()).unwrap();
    assert_eq!(tok.encode("ab<|en|>ba"), vec![6, 7, 5, 4]);
    // No byte fallback: an unmatched symbol is the unknown piece, which decodes to nothing.
    assert_eq!(tok.encode("ac"), vec![4, 3]);
    assert_eq!(tok.decode(&[6, 7, 3, 2]), "ab<|en|>");
    assert_eq!(tok.language_tag("en"), Some(7));
}

#[test]
fn every_truncation_is_an_error() {
    let bytes = small();
    for len in 0..bytes.len() {
        assert_tokenizer_error(&bytes[..len], &format!("truncated to {len}"));
    }
}

#[test]
fn bad_pieces_are_errors() {
    assert_tokenizer_error(&blob(false, &[(0.0, 9, b"a")]), "unknown type");
    assert_tokenizer_error(&blob(false, &[(0.0, 4, b"<0xZZ>")]), "bad byte piece");
    assert_tokenizer_error(&blob(false, &[(0.0, 4, b"x")]), "short byte piece");
    assert_tokenizer_error(&blob(false, &[(0.0, 0, b"\xff")]), "invalid UTF-8");
    assert_tokenizer_error(&blob(true, &[(0.0, 4, b"<0x41>")]), "missing byte pieces");
    let mut huge = small();
    huge[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_tokenizer_error(&huge, "absurd piece count");
}
