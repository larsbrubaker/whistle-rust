// Keyword biasing: on the extra clips (names, places, product words) biasing finds at least the
// measured number of keywords at one beam and at three, including every keyword the engine
// finds with its own keyword option; on the eight contract clips, irrelevant keywords leave the
// transcript exactly as it was (the contract text at one beam, the engine's at three). The bonus sweep and the counts without biasing are an ignored
// measurement.

use whistle::Options;

use crate::clips::{beams, contract_clips, extra_clips, transcribe};
use crate::support::Fixture;
use crate::wer::normalize;

/// Keywords found with biasing at the default bonus, measured 2026-10-24 (of 13), for one beam
/// and for three.
const FOUND: [(usize, usize); 2] = [(1, 9), (3, 9)];

/// Whether the keyword's words appear, in order, in the normalized transcript (case and
/// punctuation do not count: "cura" is Cura).
fn contains(text: &str, keyword: &str) -> bool {
    let words = normalize(text);
    let key = normalize(keyword);
    words.windows(key.len()).any(|w| w == key.as_slice())
}

fn with_keywords(beams: usize, keywords: &[String], bonus: f32) -> Options {
    Options {
        beams,
        keywords: keywords.to_vec(),
        keyword_bonus: bonus,
        ..Options::default()
    }
}

fn biased(n: usize, want: usize) {
    let mut missing = Vec::new();
    let mut found = 0;
    for clip in extra_clips() {
        let text = transcribe(
            &clip,
            &with_keywords(n, &clip.keywords, whistle::DEFAULT_KEYWORD_BONUS),
        )
        .text;
        println!("beams {n} {:9} {text:?}", clip.name);
        for k in &clip.keywords {
            found += usize::from(contains(&text, k));
            if contains(&clip.engine_keywords, k) && !contains(&text, k) {
                missing.push(format!(
                    "{}: {k:?} (found by the engine) not in {text:?}",
                    clip.name
                ));
            }
        }
    }
    println!("beams {n}: {found}/13 keywords");
    assert!(missing.is_empty(), "{}", missing.join("\n"));
    assert!(
        found >= want,
        "beams {n}: {found} keywords, measured {want}"
    );
}

#[test]
fn one_beam_biasing_finds_the_measured_keywords_and_every_one_the_engine_finds() {
    biased(FOUND[0].0, FOUND[0].1);
}

#[test]
fn three_beam_biasing_finds_the_measured_keywords_and_every_one_the_engine_finds() {
    biased(FOUND[1].0, FOUND[1].1);
}

fn irrelevant() -> Vec<String> {
    vec!["Siobhan".to_string(), "PETG".to_string()]
}

#[test]
fn irrelevant_keywords_leave_one_beam_on_the_contract_text() {
    for clip in contract_clips() {
        let got = transcribe(
            &clip,
            &with_keywords(1, &irrelevant(), whistle::DEFAULT_KEYWORD_BONUS),
        );
        assert_eq!(got.text, Fixture::load(&clip.name).text(), "{}", clip.name);
    }
}

/// Three beams without keywords give the engine's transcript on every contract clip (measured
/// 2026-10-24, docs/design/decoding.md), so that is the text irrelevant keywords must leave.
#[test]
fn irrelevant_keywords_leave_three_beams_on_the_engine_text() {
    for clip in contract_clips() {
        let got = transcribe(
            &clip,
            &with_keywords(3, &irrelevant(), whistle::DEFAULT_KEYWORD_BONUS),
        );
        assert_eq!(got.text, clip.engine, "{}", clip.name);
    }
}

/// The bonus strength against keywords found and contract clips changed, and the counts
/// without biasing, for docs/design/decoding.md:
/// `cargo test --release --test decoding sweep -- --ignored --nocapture`.
#[test]
#[ignore = "measurement for docs/design/decoding.md"]
fn keyword_bonus_sweep() {
    let contract = contract_clips();
    let extra = extra_clips();
    for n in [1, 3, 5] {
        let plain: usize = extra
            .iter()
            .map(|c| {
                let text = transcribe(c, &beams(n)).text;
                c.keywords.iter().filter(|k| contains(&text, k)).count()
            })
            .sum();
        println!("beams {n}: {plain}/13 keywords without biasing");
    }
    let engine = |f: fn(&crate::clips::Clip) -> &str| -> usize {
        extra
            .iter()
            .map(|c| c.keywords.iter().filter(|k| contains(f(c), k)).count())
            .sum()
    };
    println!(
        "engine: {}/13 without its keyword option, {}/13 with",
        engine(|c| &c.engine),
        engine(|c| &c.engine_keywords)
    );
    for bonus in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
        for n in [1, 3, 5] {
            let mut found = 0;
            for clip in &extra {
                let text = transcribe(clip, &with_keywords(n, &clip.keywords, bonus)).text;
                found += clip.keywords.iter().filter(|k| contains(&text, k)).count();
            }
            let changed = contract
                .iter()
                .filter(|c| {
                    transcribe(c, &with_keywords(n, &irrelevant(), bonus)).text
                        != transcribe(c, &beams(n)).text
                })
                .count();
            println!(
                "bonus {bonus} beams {n}: keywords {found}/13, contract clips changed {changed}/8"
            );
        }
    }
}
