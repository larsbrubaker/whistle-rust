// whistle.cact's tokenizer encodes and decodes exactly as upstream's `RefTokenizer`: identical
// ids on every encode case (clip texts and edge cases: accents, Polish letters, digits, inline
// language markers, byte fallback for emoji and CJK, spaces), identical round-trip strings, and
// every clip's reference token stream decodes to its manifest text.

use serde_json::Value;
use whistle::cact::Archive;
use whistle::tokenizer::{PieceType, Tokenizer};

use crate::support::archive::{manifest, model_bytes};

fn ids(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect()
}

fn tokenizer() -> Tokenizer {
    let bytes = model_bytes();
    let archive = Archive::parse(&bytes).unwrap();
    Tokenizer::from_archive(&archive).expect("whistle.cact carries a tokenizer")
}

#[test]
fn special_ids_and_piece_count_match_the_blob() {
    let tok = tokenizer();
    let m = &manifest()["tokenizer"];
    assert_eq!(tok.piece_count() as u64, m["pieces"].as_u64().unwrap());
    assert_eq!(u64::from(tok.pad_id()), m["pad_id"].as_u64().unwrap());
    assert_eq!(u64::from(tok.eos_id()), m["eos_id"].as_u64().unwrap());
    assert_eq!(u64::from(tok.bos_id()), m["bos_id"].as_u64().unwrap());
    assert_eq!(u64::from(tok.unk_id()), m["unk_id"].as_u64().unwrap());
    for (piece, id) in m["user_defined"].as_object().unwrap() {
        let id = id.as_u64().unwrap() as u32;
        assert_eq!(tok.piece(id), Some(piece.as_str()));
        assert_eq!(tok.piece_type(id), Some(PieceType::UserDefined));
    }
}

#[test]
fn language_tags_are_the_seven_languages() {
    let tok = tokenizer();
    let m = &manifest()["tokenizer"]["user_defined"];
    let tags: Vec<(&str, u32)> = tok.language_tags();
    let codes: Vec<&str> = tags.iter().map(|&(code, _)| code).collect();
    assert_eq!(codes, ["en", "de", "fr", "es", "it", "nl", "pl"]);
    for (code, id) in tags {
        assert_eq!(tok.language_tag(code), Some(id));
        assert_eq!(u64::from(id), m[format!("<|{code}|>")].as_u64().unwrap());
    }
    assert_eq!(tok.language_tag("xx"), None);
    assert_eq!(tok.language_tag("im_start"), Some(4));
}

#[test]
fn encode_matches_the_reference_on_every_case() {
    let tok = tokenizer();
    let m = manifest();
    let cases = m["tokenizer"]["encode"].as_array().unwrap();
    assert!(cases.len() >= 20, "{} encode cases", cases.len());
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let got = tok.encode(text);
        assert_eq!(got, ids(&case["ids"]), "encode {text:?}");
        assert_eq!(
            tok.decode(&got),
            case["decoded"].as_str().unwrap(),
            "decode(encode({text:?}))"
        );
    }
}

#[test]
fn every_clip_reference_stream_decodes_to_its_text() {
    let tok = tokenizer();
    let m = manifest();
    let cases = m["tokenizer"]["decode"].as_array().unwrap();
    assert_eq!(cases.len(), crate::support::clip_names().len());
    for case in cases {
        let clip = case["clip"].as_str().unwrap();
        let fixture = crate::support::Fixture::load(clip);
        let tokens = fixture.tokens();
        let text_ids = &tokens[2..tokens.len() - 1];
        assert_eq!(text_ids, ids(&case["ids"]).as_slice(), "{clip}");
        assert_eq!(tok.decode(text_ids), fixture.text(), "{clip}");
        // The whole stream too: BOS and EOS are control pieces, the tag is user-defined text.
        let tag = tok.piece(tokens[1]).unwrap().to_string();
        assert_eq!(
            tok.decode(&tokens),
            format!("{tag}{}", fixture.text()),
            "{clip}"
        );
    }
}

#[test]
fn decode_replaces_invalid_utf8_and_skips_unknown_ids() {
    let tok = tokenizer();
    let byte = |b: u8| {
        (0..tok.piece_count() as u32)
            .find(|&id| tok.piece_type(id) == Some(PieceType::Byte(b)))
            .unwrap()
    };
    // A lone continuation byte, then a truncated two-byte sequence before an ASCII byte.
    let ids = [byte(0x80), byte(b'a'), byte(0xC3), byte(b'b')];
    assert_eq!(tok.decode(&ids), "\u{FFFD}a\u{FFFD}b");
    let past_end = tok.piece_count() as u32;
    assert_eq!(
        tok.decode(&[past_end, tok.unk_id(), tok.bos_id(), byte(b'z')]),
        "z"
    );
}
