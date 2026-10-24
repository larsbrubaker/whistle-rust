// Forced languages: the language tag is fed instead of picked, and decoding continues from it.
// Forcing a clip's own language reproduces its manifest stream (the reference picked that tag
// itself); forcing another language still transcribes, in that language.

use whistle::Options;

use crate::shared::{text_tokens, wav, whistle};
use crate::support::Fixture;

#[test]
fn forcing_a_clips_own_language_gives_the_reference_tokens() {
    let f = Fixture::load("de_anna");
    let options = Options {
        language: Some("de".into()),
        ..Options::default()
    };
    let got = whistle().transcribe_wav(&wav("de_anna"), &options).unwrap();
    assert_eq!(got.tokens, text_tokens(&f.tokens()));
    assert_eq!(got.language.as_deref(), Some("de"));
    assert_eq!(got.text, f.text());
}

#[test]
fn forcing_another_language_still_transcribes_in_it() {
    let options = Options {
        language: Some("fr".into()),
        ..Options::default()
    };
    let got = whistle()
        .transcribe_wav(&wav("en_samantha"), &options)
        .unwrap();
    println!("en_samantha forced to fr: {:?}", got.text);
    assert_eq!(got.language.as_deref(), Some("fr"));
    assert!(!got.text.is_empty());
    assert!(!got.tokens.is_empty());
}
